import numpy as np
import common.setup as setup
from data.gaped import GAPED
from models.cycle_gan import Generator
from models.cycle_gan import Discriminator

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    batch_size = 1
    num_workers = 8
    dataset = GAPED()
    loader = setup.load(dataset, batch_size, num_workers)
    device = setup.device()

    in_channels = 3
    generator = setup.parallel(Generator(in_channels, 32))
    generator.to(device)
    discriminator = setup.parallel(Discriminator(in_channels, 64))
    discriminator.to(device)

    count = 0
    for batch in loader:
        image, emotion = batch
        image = image.to(device)
        emotion = emotion.to(device)

        print('before:', image.shape)
        g = generator(image)
        print('generator:', g.shape)
        d = discriminator(image)
        print('discriminator:', d.shape)

        count = min(count + batch_size, dataset.__len__())
        print('Trained', count, '/', dataset.__len__())

if __name__ == '__main__':
    main()
