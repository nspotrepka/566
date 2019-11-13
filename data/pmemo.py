import csv
from data.audio import AudioReader
from data.emotion import EmotionReader
import numpy as np
import os
import torch
from torch.utils.data import Dataset

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
    def __init__(self, size=256, audio_channels=2, offset=0, cache=False):
        self.paths = paths()
        self.index = index()
        self.static = static()
        self.read_audio = AudioReader(size, audio_channels, offset)
        self.audio = {}
        self.cache = cache
        self.read_emotion = EmotionReader()

    def __getitem__(self, i):
        key = self.index[i]
        if key in self.audio:
            audio = self.audio[key]
        else:
            audio = self.read_audio(self.paths[key])
            if self.cache:
                self.audio[key] = audio
        emotion = self.read_emotion(self.static[i])
        return audio, emotion

    def __len__(self):
        return len(self.index)
