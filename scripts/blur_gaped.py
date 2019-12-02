import common.setup as setup
from data.gaped import GAPED
import numpy as np
from skimage import io, transform
import time
from torch.utils.data import Dataset

class BlurGAPED(Dataset):
    def __init__(self):
        self.dataset1 = GAPED(size=256, cache=False, validation=False)
        self.dataset2 = GAPED(size=256, cache=False, validation=True)

    def __getitem__(self, i):
        size = self.dataset1.__len__()
        if i < size:
            image, emotion = self.dataset1.__getitem__(i)
            key = self.dataset1.names[i]
        else:
            image, emotion = self.dataset2.__getitem__(i - size)
            key = self.dataset2.names[i - size]
        path = self.dataset1.paths[key]
        image = transform.resize(image, (3, 64, 64), anti_aliasing=True)
        #image = transform.resize(image, (3, 32, 32), anti_aliasing=True)
        #image = transform.resize(image, (3, 64, 64), anti_aliasing=True)
        image = (image + 1) / 2 * 255
        image = image.astype(np.uint8)
        io.imsave(path.replace('.bmp', '.png'), image.T)
        return image, emotion

    def __len__(self):
        return self.dataset1.__len__() + self.dataset2.__len__()

def main():
    device = setup.device()

    batch_size = 8
    num_workers = 12
    dataset = BlurGAPED()
    loader = setup.load(dataset, batch_size, num_workers)

    start_time = time.time()

    count = 0
    for batch in loader:
        count = min(count + batch_size, dataset.__len__())
        print('Blurred', count, '/', dataset.__len__())

    end_time = time.time()
    print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
