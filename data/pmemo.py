import csv
import numpy as np
from pydub import AudioSegment

directory = 'data/PMEmo2019/'

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

def group(array):
    _, index = np.unique(array[:,0], return_index=True)
    return np.split(array[:,2:], index)[1:]

def pad_clip(array, shape, value):
    a = shape[0]
    b = shape[1]
    result = np.zeros(shape)
    for i in range(a):
        x = len(array[i])
        if x < b:
            result[i][:x] = array[i]
        elif x > b:
            result[i] = array[i][:b]
    return result

def index():
    array = annotation(directory + 'annotations/static_annotations.csv')
    return array[:,0]

def static():
    array = annotation(directory + 'annotations/static_annotations.csv')
    return array[:,1:]

def static_std():
    array = annotation(directory + 'annotations/static_annotations_std.csv')
    return array[:,1:]

def dynamic():
    array = annotation(directory + 'annotations/dynamic_annotations.csv')
    array = group(array)
    shape = (len(array), 30, 2)
    array = pad_clip(array, shape, 0.5)
    return np.array(array)

def dynamic_std():
    array = annotation(directory + 'annotations/dynamic_annotations_std.csv')
    array = group(array)
    array = pad_clip(array, (len(array), 30, 2), 0.0)
    return np.array(array)

def chorus_files(extension):
    id = index()
    return [directory + 'chorus/' + str(int(i)) + '.' + extension for i in id]

def chorus_mp3():
    return chorus_files('mp3')

def chorus_wav():
    return chorus_files('wav')

def convert():
    mp3 = chorus_mp3()
    wav = chorus_wav()
    for i in range(len(mp3)):
        print('Converting ' + mp3[i])
        song = AudioSegment.from_mp3(mp3[i])
        song.export(wav[i], format='wav')

def audio():
    songs = []
    for path in chorus_wav():
        print(path)
        song = AudioSegment.from_wav(path)
        samples = np.array(song.get_array_of_samples(), dtype=np.float64)
        songs.append(samples / 32768)
    array = np.array(songs)
    print("Clipping...")
    array = pad_clip(array, (len(array), 2646000), 0.0)
    print("Reshaping...")
    return array.reshape((len(array), 1323000, 2))
