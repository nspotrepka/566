import numpy as np
import common.setup as setup
from data.gaped import GAPED

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    batch_size = 8
    num_workers = 8
    dataset = GAPED()
    loader = setup.load(dataset, batch_size, num_workers)
    device = setup.device()

    count = 0
    for batch in loader:
        image, emotion = batch
        image.to(device)
        emotion.to(device)
        count = min(count + batch_size, dataset.__len__())
        print('Loaded', count, '/', dataset.__len__())

if __name__ == '__main__':
    main()
