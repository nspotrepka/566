import numpy as np
from PIL import Image
import re

directory = 'data/GAPED/'
categories = ['A', 'H', 'N', 'P', 'Sn', 'Sp']
file_suffix = '_with SD.txt'

# Get raw string annotations
def raw():
    data = []
    for category in categories:
        f = open(directory + category + file_suffix)
        lines = f.readlines()
        for line in lines:
            split = line.split()
            if len(split) > 0 and split[0] != 'Valence':
                data.append(split)
    return data

# Get file names
def names():
    names = []
    for a in raw():
        names.append(a[0].split('.')[0])
    return names

# Get emotion annotations
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

# Get image data
def images():
    n = names()
    w = 640
    h = 480
    images = np.empty((len(n), h, w, 3))
    numbers = re.compile('[0-9]+')
    for i in range(len(n)):
        name = n[i]
        category = numbers.split(name)[0]
        img = Image.open(directory + category + '/' + name + '.bmp')
        img = img.resize((w, h), Image.ANTIALIAS)
        array = np.array(img)
        if len(array.shape) == 2:
            array = np.repeat(array, 3).reshape((h, w, 3))
        images[i] = array
        img.close()
    return images
