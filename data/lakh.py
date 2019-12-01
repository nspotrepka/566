from data.midi import MidiReader
import numpy as np
import os
import torch
from torch.utils.data import Dataset

directory = 'data/clean_midi/'

def midi_paths():
    walk = os.walk(directory)
    paths = {}
    for dir in walk:
        for file in dir[2]:
            if file.endswith('.mid'):
                name = file[:file.index('.mid')]
                path = dir[0] + '/' + file
                paths[name] = path
    return paths

def paths():
    walk = os.walk(directory)
    paths = {}
    for dir in walk:
        for file in dir[2]:
            if file.endswith('.png'):
                name = file[:file.index('.png')]
                path = dir[0] + '/' + file
                paths[name] = path
    return paths

class Lakh(Dataset):
    def __init__(self, size=256, image_channels=1, cache=False,
                 validation=False):
        self.paths = paths()
        self.names = list(self.paths.keys())
        validation_cut = int(len(self.names) * 0.8)
        if validation:
            self.names = self.names[validation_cut:]
        else:
            self.names = self.names[:validation_cut]
        self.read_midi = MidiReader(size, image_channels)
        self.channels = self.read_midi.channels
        self.image = {}
        self.cache = cache

    def __getitem__(self, i):
        key = self.names[i]
        if key in self.image:
            image = self.image[key]
        else:
            image = self.read_midi(self.paths[key])
            if self.cache:
                self.image[key] = image
        emotion = torch.zeros([4]).float()
        return image, emotion

    def __len__(self):
        return len(self.names)
