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

class PMEmo(Dataset):
    rate = 32768
    full_length = 30

    def length(size):
        return int(4 * (size / 256) ** 2)

    def __init__(self, size=256, offset=0, audio_channels=2):
        self.length = PMEmo.length(size)
        assert offset >= 0 and offset < PMEmo.full_length // self.length
        self.size = size
        self.channels = 2 * audio_channels
        self.audio_channels = audio_channels
        self.paths = paths()
        self.index = index()
        self.static = static()
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain('rate', [str(PMEmo.rate)])
        self.chain.append_effect_to_chain('channels', [str(audio_channels)])
        self.chain.append_effect_to_chain('pad', ['0', str(PMEmo.full_length)])
        self.chain.append_effect_to_chain('trim',
            [str(1 + self.length * offset), str(self.length)])
        self.audio = {}

    def __getitem__(self, i):
        key = self.index[i]
        if key in self.audio:
            audio = self.audio[key]
        else:
            self.chain.set_input_file(self.paths[key])
            try:
                audio, _ = self.chain.sox_build_flow_effects()
            except RuntimeError:
                dim = [self.audio_channel, self.length * PMEmo.rate]
                audio = torch.zeros(dim)
            # This is not a great FFT size, but we get good dimensions from it
            fft_size = 2 * self.size - 1
            audio = torch.stft(audio, fft_size, fft_size, center=False)
            audio = audio.permute(0, 3, 1, 2)
            audio = audio.contiguous().view(self.channels, self.size, -1)
            self.audio[key] = audio
        emotion = torch.from_numpy(self.static[i])
        return audio.float(), emotion.float()

    def __len__(self):
        return len(self.index)
