import csv
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchaudio

directory = 'data/PMEmo2019/'

def paths():
    walk = os.walk(directory + 'chorus/')
    paths = {}
    for dir in walk:
        for file in dir[2]:
            if file.endswith('.mp3'):
                i = int(file[:file.index('.mp3')])
                path = dir[0] + file
                paths[i] = path
    return paths

def annotation(path):
    with open(path, newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=',')
        data = []
        for row in annotations:
            point = []
            for x in row:
                point.append(x)
            data.append(point)
        return np.array(data[1:], dtype=np.float64)

def index():
    array = annotation(directory + 'annotations/static_annotations.csv')
    return array[:,0]

def static_mean():
    array = annotation(directory + 'annotations/static_annotations.csv')
    return array[:,1:]

def static_std():
    array = annotation(directory + 'annotations/static_annotations_std.csv')
    return array[:,1:]

def static():
    mean = static_mean()
    std = static_std()
    return np.hstack([mean, std])

class AudioTransform:
    def __init__(self, size, audio_channels):
        self.size = size
        # This is not an intuitive FFT size, but we get good dimensions from it
        self.fft_size = 2 * size - 1
        self.audio_channels = audio_channels

    def __call__(self, audio, reverse=False):
        if reverse:
            # Unscale
            audio = audio * self.fft_size
            # Separate audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels, 2, self.size, self.size)
            # Permute dimensions
            audio = audio.permute(0, 2, 3, 1)
            # Extend time to account for samples that were clipped in FFT
            new_audio = torch.zeros(
                self.audio_channels, self.size, self.size + 1, 2)
            new_audio[:, :, :self.size, :] = audio
            # Perform inverse FFT
            audio = torchaudio.functional.istft(
                new_audio, self.fft_size, self.fft_size, center=False,
                length = 2 * self.size * self.size)
        else:
            # Perform FFT
            audio = torch.stft(
                audio, self.fft_size, self.fft_size, center=False)
            # Permute dimensions
            audio = audio.permute(0, 3, 1, 2)
            # Combine audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels * 2, self.size, self.size)
            # Scale
            audio = audio / self.fft_size
        return audio

class Audio:
    rate = 32768
    full_length = 30

    def __init__(self, size=256, audio_channels=2):
        self.length = Audio.length(size)
        self.transform = AudioTransform(size, audio_channels)

    # Get length in seconds
    def length(size):
        return int(4 * (size / 256) ** 2)

class AudioReader(Audio):
    def __init__(self, size=256, audio_channels=2, offset=0):
        super(AudioReader, self).__init__(size, audio_channels)
        assert offset >= 0 and offset < Audio.full_length // self.length
        cue = offset * self.length
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain('rate', [str(Audio.rate)])
        self.chain.append_effect_to_chain('channels', [str(audio_channels)])
        self.chain.append_effect_to_chain('pad', ['0', str(Audio.full_length)])
        self.chain.append_effect_to_chain('trim', [str(cue), str(self.length)])
        self.dim = [audio_channels, self.length * Audio.rate]

    def __call__(self, path):
        self.chain.set_input_file(path)
        try:
            audio, _ = self.chain.sox_build_flow_effects()
        except RuntimeError:
            audio = torch.zeros(self.dim)
        audio = self.transform(audio)
        audio = audio.float()
        return audio

class PMEmo(Dataset):
    def __init__(self, size=256, audio_channels=2, offset=0, cache=False):
        self.paths = paths()
        self.index = index()
        self.static = static()
        self.reader = AudioReader(size, audio_channels, offset)
        self.audio = {}
        self.cache = cache

    def __getitem__(self, i):
        key = self.index[i]
        if key in self.audio:
            audio = self.audio[key]
        else:
            audio = self.reader(self.paths[key])
            if self.cache:
                self.audio[key] = audio
        emotion = self.static[i]
        emotion = torch.from_numpy(emotion).float()
        return audio, emotion

    def __len__(self):
        return len(self.index)
