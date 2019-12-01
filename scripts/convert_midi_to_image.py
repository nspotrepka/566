import json
from music21 import converter, instrument, note, chord
import numpy as np
from skimage import io
import sys
import warnings

def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))

        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element.notes:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {'start': start, 'pitch': notes, 'dur': durations}

def midi2image(midi_path):
    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            if instrument_i.partName is None:
                data['instrument_{}'.format(i)] = get_notes(notes_to_parse)
                i+=1
            else:
                data[instrument_i.partName] = get_notes(notes_to_parse)

    except:
        notes_to_parse = mid.flat.notes
        data['instrument_0'.format(i)] = get_notes(notes_to_parse)

    resolution = 0.25

    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
        upperBoundNote = 128
        lowerBoundNote = 0
        maxSongLength = 128

        index = 0
        prev_index = 0
        repetitions = 0
        while repetitions < 1:
            if prev_index >= len(values['pitch']):
                break

            matrix = np.zeros((upperBoundNote - lowerBoundNote, maxSongLength))

            pitchs = values['pitch']
            durs = values['dur']
            starts = values['start']

            for i in range(prev_index, len(pitchs)):
                pitch = pitchs[i]

                dur = int(durs[i] / resolution)
                start = int(starts[i] / resolution)

                if dur + start - index * maxSongLength < maxSongLength:
                    for j in range(start, start + dur):
                        new_j = j - index * maxSongLength
                        if new_j >= 0:
                            matrix[pitch - lowerBoundNote, new_j] = 255
                else:
                    prev_index = i
                    break

            if matrix.max() != matrix.min():
                matrix = matrix.astype(np.uint8)
                formatted = f'_{instrument_name}_{index}.png'
                path = midi_path.replace('.mid', formatted)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    io.imsave(path, matrix)
            index += 1
            repetitions += 1

if __name__ == '__main__':
    midi_path = sys.argv[1]
    midi2image(midi_path)
