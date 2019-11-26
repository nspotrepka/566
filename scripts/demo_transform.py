from argparse import ArgumentParser
import common.setup as setup
from data.audio import AudioReader
from data.audio import AudioWriter
from data.image import ImageReader
from data.image import ImageWriter
from models.cyclegan.model import CycleGAN
import os
import torch

def main(params):
    """
    Run transform forward and backward.
    """
    size = params.data_size
    image_path = params.image
    audio_path = params.audio

    image_channels = 3
    audio_channels = 2
    ext_image = '.png'
    ext_audio = '.wav'

    # Perform image to audio
    if image_path is not None:
        dir = os.path.dirname(image_path)
        base, _ = os.path.splitext(os.path.basename(image_path))

        print('Reading image...')
        new_image_path = os.path.join(dir, 'transform_' + base + ext_image)
        read_image = ImageReader(size, image_channels)
        image = read_image(image_path)

        print('Writing image...')
        write_image = ImageWriter(size, image_channels)
        write_image(new_image_path, image)

    # Perform audio to image
    if audio_path is not None:
        dir = os.path.dirname(audio_path)
        base, _ = os.path.splitext(os.path.basename(audio_path))

        print('Reading audio...')
        new_audio_path = os.path.join(dir, 'transform_' + base + ext_audio)
        new_image_path = os.path.join(dir, 'transform_' + base + ext_image)
        setup.init_audio()
        read_audio = AudioReader(size, audio_channels)
        audio = read_audio(audio_path)
        setup.shutdown_audio()

        print('Writing audio...')
        write_audio = AudioWriter(size, audio_channels)
        write_audio(new_audio_path, audio)

        print('Writing image...')
        write_image = ImageWriter(size, image_channels)
        write_image(new_image_path, audio)

    # Done
    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image', help='path to image')
    parser.add_argument('--audio', help='path to audio')
    parser.add_argument('--data_size', type=int, default=256, help='size of converted audio/image')

    params = parser.parse_args()

    main(params)
