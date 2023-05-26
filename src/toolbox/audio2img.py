import librosa
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import torchvision
import random


class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig, sig])

        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
          # Nothing to do
          return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
    
    
    
def audio_to_img(path,
                 period= 5,
                 sr= None,
                 n_mels= 128,
                 n_fft= None,
                 hop_length= None,
                 fmin = 20,
                 fmax = 16000):
    
    def mono_to_color(X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)
        return V
    
    if type(path)==str:
        data, sr = librosa.load(path, sr= sr)
        max_sec= int( len(data)//sr )+1

        ## split audio by period
        datas = [ data[i * sr: (i+period) * sr] for i in range(0, max_sec, period) ]
        if len(datas[-1])<sr*period: datas[-1]= list(datas[-1]) + [0]*( sr*period-len(datas[-1]) )
    else:
        datas= [path]
    
    ## audio to melspectrogram
    melspec = librosa.feature.melspectrogram(y= np.array(datas), ## drop last
                                             sr= sr, 
                                             n_mels= n_mels, 
                                             n_fft= n_fft,
                                             hop_length= hop_length,
                                             fmin= fmin, 
                                             fmax= fmax)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    img= np.array([mono_to_color(im) for im in melspec])
            
    return img