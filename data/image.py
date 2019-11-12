import numpy as np
import os
import torch
from torch.utils.data import Dataset
import re
from skimage import io, transform, util

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

class ImageTransform:
    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels
        self.padding = (self.width - self.height) // 2

    def __call__(self, image, reverse=False):
        if reverse:
            # Unscale
            image = (image + 1) / 2
            # Crop height
            # Skip to observe artifacts
            # image = image[:, self.padding:self.padding + self.height, :]
            # Transpose dimensions
            image = image.T
        else:
            # Resize width and height
            image = transform.resize(image, (self.width, self.height))
            # Transpose dimensions
            image = image.T
            # Pad channels and height
            image = util.pad(image,
                ((0, self.channels), (self.padding, self.padding), (0, 0)),
                mode='reflect')
            # Crop channels
            image = image[:self.channels, :, :]
            # Scale
            image = image * 2 - 1
        return image

class Image:
    def __init__(self, size=256, image_channels=3):
        self.transform = ImageTransform(size, size * 3 // 4, image_channels)

class ImageReader(Image):
    def __init__(self, size=256, image_channels=3):
        super(ImageReader, self).__init__(size, image_channels)

    def __call__(self, path):
        image = io.imread(path)
        image = self.transform(image)
        image = torch.from_numpy(image).float()
        return image

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
