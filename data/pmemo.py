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
    rate = 44100
    channels = 2
    length = 4

    def __init__(self, size=512):
        self.size = size
        self.paths = paths()
        self.index = index()
        self.static = static()
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain('rate', [str(PMEmo.rate)])
        self.chain.append_effect_to_chain('channels', [str(PMEmo.channels)])
        self.chain.append_effect_to_chain('pad', ['0', str(PMEmo.length)])
        self.chain.append_effect_to_chain('trim', ['0', str(PMEmo.length)])
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
                audio = torch.zeros([PMEmo.channels, PMEmo.length * PMEmo.rate])
            audio = torch.stft(audio, self.size-1, self.size-1, center=False)
            self.audio[key] = audio
        emotion = torch.from_numpy(self.static[i])
        return audio.float(), emotion.float()

    def __len__(self):
        return len(self.index)
