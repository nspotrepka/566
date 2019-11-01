import common.setup as setup
from data.pmemo import PMEmo
import time
import torchaudio

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    device = setup.device()

    batch_size = 8
    dataset = PMEmo()
    loader = setup.load(dataset, batch_size)

    torchaudio.initialize_sox()

    for _ in range(2):
        start_time = time.time()

        count = 0
        for batch in loader:
            audio, emotion = batch
            audio.to(device)
            emotion.to(device)
            print(audio.shape)
            print(emotion.shape)
            count = min(count + batch_size, dataset.__len__())
            print('Loaded', count, '/', dataset.__len__())

        end_time = time.time()
        print('Time:', end_time - start_time, 'sec')

    torchaudio.shutdown_sox()

if __name__ == '__main__':
    main()
