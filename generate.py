from argparse import ArgumentParser
from argparse import Namespace
import common.setup as setup
from data.audio import AudioReader
from data.audio import AudioWriter
from data.image import ImageReader
from data.image import ImageWriter
from models.cyclegan.model import CycleGAN
import torch

def main(params):
    size = params.data_size

    # Load model from checkpoint
    print('Loading model...')
    map_location = lambda storage, loc: storage
    checkpoint = torch.load(params.checkpoint_path, map_location=map_location)
    try:
        checkpoint_hparams = checkpoint['hparams']
        image_channels = checkpoint_hparams['in_channels']
        audio_channels = checkpoint_hparams['out_channels'] // 2
        model = CycleGAN(None, **checkpoint_hparams)
    except KeyError:
        print('Warning: No hyperparameters found. Using defaults.')
        image_channels = 3
        audio_channels = 2
        model = CycleGAN(None, image_channels, audio_channels * 2)
    model.load_state_dict(checkpoint['state_dict'])
    model.on_load_checkpoint(checkpoint)
    model.eval()
    model.freeze()

    write_image = ImageWriter(size, image_channels)
    write_audio = AudioWriter(size, audio_channels)

    # Perform image to audio
    if params.image_path is not None:
        print('Reading image...')
        read_image = ImageReader(size, image_channels)
        image = read_image(params.image_path)
        image = torch.unsqueeze(image, 0)
        if params.fake_audio_path is not None:
            print('Generating fake audio...')
            fake_audio = model.gen_a_to_b(image)
            write_audio(params.fake_audio_path, torch.squeeze(fake_audio, 0))
        if params.cycle_image_path is not None:
            print('Generating cycle image...')
            cycle_image = model.gen_b_to_a(fake_audio)
            write_image(params.cycle_image_path, torch.squeeze(cycle_image, 0))

    # Perform audio to image
    if params.audio_path is not None:
        print('Reading audio...')
        setup.init_audio()
        read_audio = AudioReader(size, audio_channels)
        audio = read_audio(params.audio_path)
        setup.shutdown_audio()
        audio = torch.unsqueeze(audio, 0)
        write_audio("data/audio/original.wav", torch.squeeze(audio, 0))
        if params.fake_image_path is not None:
            print('Generating fake image...')
            fake_image = model.gen_b_to_a(audio)
            write_image(params.fake_image_path, torch.squeeze(fake_image, 0))
        if params.cycle_audio_path is not None:
            print('Generating cycle audio...')
            cycle_audio = model.gen_a_to_b(fake_image)
            write_audio(params.cycle_audio_path, torch.squeeze(cycle_audio, 0))

    # Done
    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', help='path to model checkpoint')
    parser.add_argument('--image_path', help='path to image')
    parser.add_argument('--audio_path', help='path to audio')
    parser.add_argument('--fake_image_path', help='path to save fake image')
    parser.add_argument('--fake_audio_path', help='path to save fake audio')
    parser.add_argument('--cycle_image_path', help='path to save cycle image')
    parser.add_argument('--cycle_audio_path', help='path to save cycle audio')
    parser.add_argument('--data_size', type=int, default=128, help='size of converted audio/image')

    params = parser.parse_args()

    if params.checkpoint_path is None:
        print('Please provide a checkpoint path')
        parser.print_help()

    main(params)
