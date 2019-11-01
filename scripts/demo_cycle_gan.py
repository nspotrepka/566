import numpy as np
import common.setup as setup
from data.gaped import GAPED
from models.cycle_gan import CycleGAN
import time

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
    out_channels = 3
    model = setup.parallel(CycleGAN(in_channels, out_channels, 8, 16))
    model = model.to(device)

    count = 0
    for batch in loader:
        start_time = time.time()

        image, emotion = batch
        image = image.to(device)
        emotion = emotion.to(device)

        model.train(image, image)

        end_time = time.time()

        count = min(count + batch_size, dataset.__len__())
        print('Trained', count, '/', dataset.__len__())
        print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
