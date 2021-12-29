import os
if os.getcwd() != '/content/tacotron2':
    os.chdir('tacotron2')
import time
import argparse
import math
import numpy as np
import gc
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import to_gpu
import random
import time
from math import e

from tqdm import tqdm_notebook as tqdm
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
import gradient_adaptive_factor
from text import text_to_sequence
from distutils.dir_util import copy_tree
import matplotlib.pylab as plt
from scipy.io.wavfile import read

def create_mels():
    print("Generating Mels")
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)
    def save_mel(file):
        audio, sampling_rate = load_wav_to_torch(file)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(file, 
                sampling_rate, stft.sampling_rate))
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).cpu().numpy()
        np.save(file.replace('.wav', ''), melspec)

    import glob
    wavs = glob.glob('wavs/out/*.wav')
    for i in tqdm(wavs):
        save_mel(i)
    stft = None


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    #dist.all_reduce(rt, op=dist.ReduceOp.SUM) # Updated ReduceOp, needs to be tested
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def plot_alignment(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                   interpolation='none')
    ax.autoscale(enable=True, axis="y", tight=True)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, epoch, start_eposh, learning_rate):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Epoch: {} Validation loss {}: {:9f}  Time: {:.1f}m LR: {:.6f}".format(epoch, iteration, reduced_val_loss,(time.perf_counter()-start_eposh)/60, learning_rate))
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)
        if hparams.show_alignments:
            _, _, mel_outputs, gate_outputs, alignments = y_pred
            idx = random.randint(0, alignments.size(0) - 1)
            plot_alignment(alignments[idx].data.cpu().numpy().T,"Validation Loss: "+str(reduced_val_loss))

def calculate_global_mean(data_loader, global_mean_npy):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return to_gpu(torch.tensor(global_mean))
    sums = []
    frames = []
    print('calculating global mean...')
    for i, batch in enumerate(data_loader):
        (text_padded, input_lengths, mel_padded, gate_padded,
         output_lengths, ctc_text, ctc_text_lengths) = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
    #if global_mean_npy:
    #    np.save(global_mean_npy, global_mean.cpu().numpy())
    print('Done')
    return to_gpu((sum(sums) / sum(frames)).float())

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, log_directory2):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    if hparams.drop_frame_rate > 0.:
        global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy)
        hparams.global_mean = global_mean
    gc.collect()

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    gc.collect()
    
    start_eposh = time.perf_counter()
    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, hparams.epochs),desc="Epoch: "):
        gc.collect(); print("\nStarting Epoch: {} Iteration: {}".format(epoch, iteration))
        start_eposh = time.perf_counter() # eposh is russian, not a typo
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            start = time.perf_counter()
            if iteration < decay_start: learning_rate = A_
            else: iteration_adjusted = iteration - decay_start; learning_rate = (A_*(e**(-iteration_adjusted/B_))) + C_
            learning_rate = max(min_learning_rate, learning_rate) # output the largest number
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if model.mi is not None:
                # transpose to [b, T, dim]
                decoder_outputs = y_pred[0].transpose(2, 1)
                ctc_text, ctc_text_lengths, aco_lengths = x[-2], x[-1], x[4]
                taco_loss = loss
                mi_loss = model.mi(decoder_outputs, ctc_text, aco_lengths, ctc_text_lengths)
                if hparams.use_gaf:
                    if i % gradient_adaptive_factor.UPDATE_GAF_EVERY_N_STEP == 0:
                        safe_loss = 0. * sum([x.sum() for x in model.parameters()])
                        gaf = gradient_adaptive_factor.calc_grad_adapt_factor(
                            taco_loss + safe_loss, mi_loss + safe_loss, model.parameters(), optimizer)
                        gaf = min(gaf, hparams.max_gaf)
                else:
                    gaf = 1.0
                loss = loss + gaf * mi_loss
            else:
                taco_loss = loss
                mi_loss = torch.tensor([-1.0])
                gaf = -1.0

            if hparams.distributed_run: reduced_loss = reduce_tensor(loss.data, n_gpus).item(); taco_loss = reduce_tensor(taco_loss.data, n_gpus).item(); mi_loss = reduce_tensor(mi_loss.data, n_gpus).item()
            else: reduced_loss = loss.item(); taco_loss = taco_loss.item(); mi_loss = mi_loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: loss.backward()

            if hparams.fp16_run: grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hparams.grad_clip_thresh);  is_overflow = math.isnan(grad_norm)
            else: grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                logger.log_training(
                    reduced_loss, taco_loss, mi_loss, grad_norm, gaf, learning_rate, duration, iteration)
                #print("Batch {} loss {:.6f} Grad Norm {:.6f} Time {:.6f}".format(iteration, reduced_loss, grad_norm, duration), end='\r', flush=True)

            iteration += 1
        validate(model, criterion, valset, iteration,
                 hparams.batch_size, n_gpus, collate_fn, logger,
                 hparams.distributed_run, rank, epoch, start_eposh, learning_rate)
        save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)
        if log_directory2 != None:
            copy_tree(log_directory, log_directory2)

warm_start=False
n_gpus=1
rank=0
group_name=None

hparams = create_hparams()

# hparams to Tune
gradient_adaptive_factor.UPDATE_GAF_EVERY_N_STEP = 10
hparams.use_mmi=True
hparams.use_gaf=True
hparams.max_gaf=0.5
hparams.drop_frame_rate = 0.0
hparams.p_teacher_forcing=1.0 # not working right now

# Dropout                   # https://pytorch.org/assets/images/tacotron2_diagram.png <-- orange = decoder
hparams.p_attention_dropout=0.1
hparams.p_decoder_dropout=0.1

# Learning Rate             # https://www.desmos.com/calculator/ptgcz4vzsw / http://boards.4channel.org/mlp/thread/34778298#p34789030
decay_start = 15000         # wait till decay_start to start decaying learning rate
A_ = 5e-4                   # Start/Max Learning Rate
B_ = 8000                   # Decay Rate
C_ = 0                      # Shift learning rate equation by this value
min_learning_rate = 1e-5    # Min Learning Rate

# Quality of Life
model_filename = 'current_model'
generate_mels = 0
hparams.show_alignments = 0

# Audio Parameters
hparams.sampling_rate=48000
hparams.filter_length=2400
hparams.hop_length=600
hparams.win_length=2400
hparams.n_mel_channels=80
hparams.mel_fmin=0.0
hparams.mel_fmax=18000.0

hparams.batch_size = 6
hparams.load_mel_from_disk = True
hparams.training_files = "filelists/clipper_train_filelist.txt"
hparams.validation_files = "filelists/clipper_val_filelist.txt"
hparams.ignore_layers = []
hparams.epochs = 1

torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
print('FP16 Run:', hparams.fp16_run)
print('Dynamic Loss Scaling:', hparams.dynamic_loss_scaling)
print('Distributed Run:', hparams.distributed_run)
print('cuDNN Enabled:', hparams.cudnn_enabled)
print('cuDNN Benchmark:', hparams.cudnn_benchmark)

output_directory = '/content/drive/My Drive/colab/outdir'
log_directory = '/content/tacotron2/logs'
checkpoint_path = output_directory+(r'/')+model_filename
log_directory2 = '/content/drive/My Drive/colab/logs'

if generate_mels:
    create_mels(); gc.collect()
train(output_directory, log_directory, checkpoint_path,
      warm_start, n_gpus, rank, group_name, hparams, log_directory2)
