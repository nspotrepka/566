import common.common as common
import data.pmemo as pmemo
import numpy as np

def main():
    a = pmemo.static()
    b = pmemo.static_std()
    c = pmemo.dynamic()
    d = pmemo.dynamic_std()

    print('Loading songs...')
    songs = pmemo.chorus()

    print('number of songs = ' + str(len(songs)))

    lengths = [len(s) for s in songs]
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    min_id = int(a[np.argmin(lengths)][0])
    max_id = int(a[np.argmax(lengths)][0])

    print('min song length in samples = ' + str(min_length) + ', id = ' + str(min_id))
    print('max song length in samples = ' + str(max_length) + ', id = ' + str(max_id))

if __name__ == "__main__":
    main()
