import common.setup as setup
from data.gaped import GAPED
import time

def main():
    device = setup.device()

    batch_size = 8
    dataset = GAPED(size=128, cache=True)
    loader = setup.load(dataset, batch_size)

    for epoch in range(2):
        start_time = time.time()

        count = 0
        for batch in loader:
            image, emotion = batch
            image.to(device)
            emotion.to(device)
            if epoch == 0 and count == 0:
                print(image.shape)
                print(emotion.shape)
            count = min(count + batch_size, dataset.__len__())
            print('Loaded', count, '/', dataset.__len__())

        end_time = time.time()
        print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
