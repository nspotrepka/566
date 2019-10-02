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
    unique, index = np.unique(array[:,0], return_index=True)
    array = np.split(array[:,1:], index)[1:]
    return [[unique[i], array[i]] for i in range(len(unique))]

def static():
    array = annotation(directory + 'annotations/static_annotations.csv')
    return array

def static_std():
    array = annotation(directory + 'annotations/static_annotations_std.csv')
    return array

def dynamic():
    array = annotation(directory + 'annotations/dynamic_annotations.csv')
    return group(array)

def dynamic_std():
    array = annotation(directory + 'annotations/dynamic_annotations_std.csv')
    return group(array)

def chorus_files(extension):
    id = static()[:,0]
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

def chorus():
    songs = []
    for path in chorus_wav():
        song = AudioSegment.from_wav(path)
        samples = np.array(song.get_array_of_samples(), dtype=np.float64)
        songs.append(samples / 32768)
    return np.array(songs)
