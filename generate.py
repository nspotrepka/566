from argparse import ArgumentParser
from models.cyclegan.model import CycleGAN

def main(params):
    # Load model from checkpoint
    print('Loading model...')
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path=parser.checkpoint_path
    )
    model.eval()
    model.freeze()

    size = params.data_size
    image_channels = params.image_channels
    audio_channels = params.audio_channels

    write_image = ImageWriter(size, image_channels)
    write_audio = AudioWriter(size, audio_channels)

    # Perform image to audio
    if params.image_path is not None:
        print('Reading image...')
        read_image = ImageReader(size, image_channels)
        image = read_image(params.image_path)
        if params.fake_audio_path is not None:
            print('Generating fake audio...')
            fake_audio = model.gen_a_to_b(image)
            write_audio(params.fake_audio_path, fake_audio)
        if params.cycle_image_path is not None:
            print('Generating cycle image...')
            cycle_image = model.gen_b_to_a(fake_audio)
            write_image(params.cycle_image_path, cycle_image)

    # Perform audio to image
    if params.audio_path is not None:
        print('Reading audio...')
        read_audio = AudioReader(size, audio_channels)
        audio = read_audio(params.audio_path)
        if params.fake_image_path is not None:
            print('Generating fake image...')
            fake_image = model.gen_b_to_a(image)
            write_image(params.fake_image_path, fake_image)
        if params.cycle_audio_path is not None:
            print('Generating cycle audio...')
            cycle_audio = model.gen_a_to_b(fake_image)
            write_audio(params.cycle_audio_path, cycle_audio)

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
    parser.add_argument('--image_channels', type=int, default=3, help='number of channels to use in source image')
    parser.add_argument('--audio_channels', type=int, default=2, help='number of channels to use in source audio')

    params = parser.parse_args()

    if params.checkpoint_path is None:
        print('Please provide a checkpoint path')
        parser.print_help()

    main(params)
