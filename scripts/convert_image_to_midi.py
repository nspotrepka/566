import numpy as np
from music21 import instrument, note, chord, stream

lowerBoundNote = 0
resolution = 0.25

def column2notes(column):
    notes = []
    for i in range(len(column)):
        if column[i] > 127.5:
            notes.append(i + lowerBoundNote)
    return notes

def updateNotes(newNotes,prevNotes):
    res = {}
    for note in newNotes:
        if note in prevNotes:
            res[note] = prevNotes[note] + resolution
        else:
            res[note] = resolution
    return res

def image2midi(im_arr, path):
    offset = 0
    output_notes = []

    prev_notes = updateNotes(im_arr.T[0, :], {})
    for column in im_arr.T[1:, :]:
        notes = column2notes(column)
        # pattern is a chord
        notes_in_chord = notes
        old_notes = prev_notes.keys()
        for old_note in old_notes:
            if not old_note in notes_in_chord:
                new_note = note.Note(old_note,
                                     quarterLength=prev_notes[old_note])
                new_note.storedInstrument = instrument.Piano()
                if offset - prev_notes[old_note] >= 0:
                    new_note.offset = offset - prev_notes[old_note]
                    output_notes.append(new_note)
                elif offset == 0:
                    new_note.offset = offset
                    output_notes.append(new_note)
                else:
                    print(offset,prev_notes[old_note],old_note)

        prev_notes = updateNotes(notes_in_chord,prev_notes)

        # increase offset each iteration so that notes do not stack
        offset += resolution

    for old_note in prev_notes.keys():
        new_note = note.Note(old_note, quarterLength=prev_notes[old_note])
        new_note.storedInstrument = instrument.Piano()
        new_note.offset = offset - prev_notes[old_note]

        output_notes.append(new_note)

    prev_notes = updateNotes(notes_in_chord, prev_notes)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=path)

if __name__ == '__main__':
    image_path = sys.argv[1]
    image2midi(image_path)
