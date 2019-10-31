import numpy as np
import os
import torch
from torch.utils.data import Dataset
import re
from skimage import io, transform

directory = 'data/GAPED/'
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

class Transform(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        image = image[:,:,:3]
        image = transform.resize(image, (self.width, self.height))
        image = image.T
        return image

class GAPED(Dataset):
    width = 640
    height = 480

    def __init__(self):
        self.paths = paths()
        self.names = names()
        self.emotion = emotion()
        self.transform = Transform(GAPED.width, GAPED.height)

    def __getitem__(self, i):
        image = io.imread(self.paths[self.names[i]])
        image = self.transform(image)
        image = torch.from_numpy(image)
        emotion = torch.from_numpy(self.emotion[i])
        return image.float(), emotion.float()

    def __len__(self):
        return len(self.names)
