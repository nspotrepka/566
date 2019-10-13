import data.gaped as gaped
import numpy as np

def main():
    emotion = gaped.emotion()
    images = gaped.images()

    print('emotion = ' + str(emotion.shape))
    print('images = ' + str(images.shape))

if __name__ == "__main__":
    main()
