import data.pmemo as pmemo
import numpy as np

def main():
    a = pmemo.static()
    b = pmemo.static_std()
    c = pmemo.dynamic()
    d = pmemo.dynamic_std()
    songs = pmemo.audio()

    print('static = ' + str(a.shape))
    print('static_std = ' + str(b.shape))
    print('dynamic = ' + str(c.shape))
    print('dynamic_std = ' + str(d.shape))
    print('audio = ' + str(songs.shape))

if __name__ == "__main__":
    main()
