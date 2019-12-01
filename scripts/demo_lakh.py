import common.setup as setup
from data.lakh import Lakh
import time

def main():
    device = setup.device()

    batch_size = 8
    dataset = Lakh(size=128, cache=True)
    loader = setup.load(dataset, batch_size)

    for epoch in range(2):
        start_time = time.time()

        count = 0
        for batch in loader:
            image, _ = batch
            image.to(device)
            if epoch == 0 and count == 0:
                print(image.shape)
            count = min(count + batch_size, dataset.__len__())
            print('Loaded', count, '/', dataset.__len__())

        end_time = time.time()
        print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
