from data.image import ImageReader
import numpy as np
import os
import torch
from torch.utils.data import Dataset

directory = 'data/GAPED/GAPED/'
categories = ['A', 'H', 'N', 'P', 'Sn', 'Sp']
file_suffix = '_with SD.txt'

def paths():
    walk = os.walk(directory)
    paths = {}
    for dir in walk:
        for file in dir[2]:
            if file.endswith('.bmp'):
                name = file[:file.index('.bmp')]
                path = dir[0] + '/' + file
                paths[name] = path
    return paths

def raw():
    data = []
    for category in categories:
        f = open(directory + category + file_suffix)
        lines = f.readlines()
        for i in range(len(lines)):
            if i > 0:
                line = lines[i]
                split = line.split()
                if len(split) > 0:
                    data.append(split)
    return data

def names():
    names = []
    for a in raw():
        names.append(a[0].split('.')[0])
    return names

def emotion():
    data = []
    for a in raw():
        row = []
        row.append(float(a[3]) / 100)
        row.append(float(a[1]) / 100)
        row.append(float(a[4][1:-1]) / 100)
        row.append(float(a[2][1:-1]) / 100)
        data.append(row)
    return np.array(data)

class GAPED(Dataset):
    def __init__(self, size=256, image_channels=3, cache=False):
        self.paths = paths()
        self.names = names()
        self.emotion = emotion()
        self.reader = ImageReader(size, image_channels)
        self.image = {}
        self.cache = cache

    def __getitem__(self, i):
        key = self.names[i]
        if key in self.image:
            image = self.image[key]
        else:
            image = self.reader(self.paths[self.names[i]])
            if self.cache:
                self.image[key] = image
        emotion = self.emotion[i]
        emotion = torch.from_numpy(emotion).float()
        return image, emotion

    def __len__(self):
        return len(self.names)
