import common.setup as setup
from data.pmemo import PMEmo
import os
import time

def main():
    # This is an unsafe, unsupported, undocumented workaround
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    device = setup.device()

    batch_size = 8
    dataset = PMEmo(size=128, cache=True)
    loader = setup.load(dataset, batch_size)

    setup.init_audio()

    for epoch in range(2):
        start_time = time.time()

        count = 0
        for batch in loader:
            audio, emotion = batch
            audio.to(device)
            emotion.to(device)
            if epoch == 0 and count == 0:
                print(audio.shape)
                print(emotion.shape)
            count = min(count + batch_size, dataset.__len__())
            print('Loaded', count, '/', dataset.__len__())

        end_time = time.time()
        print('Time:', end_time - start_time, 'sec')

    setup.shutdown_audio()

if __name__ == '__main__':
    main()
