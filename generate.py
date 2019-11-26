from argparse import ArgumentParser
from argparse import Namespace
import common.setup as setup
from data.audio import AudioReader
from data.audio import AudioWriter
from data.image import ImageReader
from data.image import ImageWriter
from models.cyclegan.model import CycleGAN
import os
import torch

def main(params):
    size = params.data_size
    image_path = params.image
    audio_path = params.audio

    # Load model from checkpoint
    print('Loading model...')
    map_location = lambda storage, loc: storage
    checkpoint = torch.load(params.checkpoint, map_location=map_location)
    try:
        checkpoint_hparams = checkpoint['hparams']
        image_channels = checkpoint_hparams['in_channels']
        audio_channels = checkpoint_hparams['out_channels']
        model = CycleGAN(None, None, **checkpoint_hparams)
    except KeyError:
        print('Warning: No hyperparameters found. Using defaults.')
        image_channels = 3
        audio_channels = 2
        model = CycleGAN(None, None, image_channels, audio_channels)
    model.load_state_dict(checkpoint['state_dict'])
    model.on_load_checkpoint(checkpoint)
    model.eval()
    model.freeze()

    write_image = ImageWriter(size, image_channels)
    write_audio = AudioWriter(size, audio_channels)
    ext_image = '.png'
    ext_audio = '.wav'

    # Perform image to audio
    if image_path is not None:
        dir = os.path.dirname(image_path)
        base, _ = os.path.splitext(os.path.basename(image_path))

        print('Reading image...')
        original_image_path = os.path.join(dir, 'original_' + base + ext_image)
        read_image = ImageReader(size, image_channels)
        image = read_image(image_path)
        image = torch.unsqueeze(image, 0)
        write_image(original_image_path, torch.squeeze(image, 0))

        print('Generating fake audio...')
        fake_image_path = os.path.join(dir, 'fake_' + base + ext_image)
        fake_audio_path = os.path.join(dir, 'fake_' + base + ext_audio)
        fake_audio = model.gen_a_to_b(image)
        fake_audio_out = torch.squeeze(fake_audio, 0)
        write_image(fake_image_path, fake_audio_out)
        write_audio(fake_audio_path, fake_audio_out)

        print('Generating cycle image...')
        cycle_image_path = os.path.join(dir, 'cycle_' + base + ext_image)
        cycle_image = model.gen_b_to_a(fake_audio)
        write_image(cycle_image_path, torch.squeeze(cycle_image, 0))

        print('Generating diff image...')
        diff_image_path = os.path.join(dir, 'diff_' + base + ext_image)
        diff_image = cycle_image - image
        write_image(diff_image_path, torch.squeeze(diff_image, 0))

    # Perform audio to image
    if audio_path is not None:
        dir = os.path.dirname(audio_path)
        base, _ = os.path.splitext(os.path.basename(audio_path))

        print('Reading audio...')
        original_image_path = os.path.join(dir, 'original_' + base + ext_image)
        original_audio_path = os.path.join(dir, 'original_' + base + ext_audio)
        setup.init_audio()
        read_audio = AudioReader(size, audio_channels)
        audio = read_audio(audio_path)
        setup.shutdown_audio()
        audio = torch.unsqueeze(audio, 0)
        audio_out = torch.squeeze(audio, 0)
        write_image(original_image_path, audio_out)
        write_audio(original_audio_path, audio_out)

        print('Generating fake image...')
        fake_image_path = os.path.join(dir, 'fake_' + base + ext_image)
        fake_image = model.gen_b_to_a(audio)
        write_image(fake_image_path, torch.squeeze(fake_image, 0))

        print('Generating cycle audio...')
        cycle_image_path = os.path.join(dir, 'cycle_' + base + ext_image)
        cycle_audio_path = os.path.join(dir, 'cycle_' + base + ext_audio)
        cycle_audio = model.gen_a_to_b(fake_image)
        cycle_audio_out = torch.squeeze(cycle_audio, 0)
        write_image(cycle_image_path, cycle_audio_out)
        write_audio(cycle_audio_path, cycle_audio_out)

        print('Generating diff audio...')
        diff_image_path = os.path.join(dir, 'diff_' + base + ext_image)
        diff_audio_path = os.path.join(dir, 'diff_' + base + ext_audio)
        diff_audio = cycle_audio - audio
        diff_audio_out = torch.squeeze(diff_audio, 0)
        write_image(diff_image_path, diff_audio_out)
        write_audio(diff_audio_path, diff_audio_out)

    # Done
    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument('--data_size', type=int, default=256, help='size of converted image or audio')
    required.add_argument('--checkpoint', help='path to model checkpoint', required=True)
    optional.add_argument('--image', help='path to image')
    optional.add_argument('--audio', help='path to audio')
    parser._action_groups.append(optional)

    params = parser.parse_args()
    main(params)
