import data.pmemo as pmemo
import numpy as np

def main():
    static = pmemo.static()
    dynamic = pmemo.dynamic()
    audio = pmemo.audio()

    print('static = ' + str(static.shape))
    print('dynamic = ' + str(dynamic.shape))
    print('audio = ' + str(audio.shape))

if __name__ == "__main__":
    main()
