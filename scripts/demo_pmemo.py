import common.setup as setup
from data.pmemo import PMEmo
import time

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    # This is an unsafe, unsupported, undocumented workaround
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    device = setup.device()

    batch_size = 8
    dataset = PMEmo(256)
    loader = setup.load(dataset, batch_size)

    setup.init_audio()

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

    setup.shutdown_audio()

if __name__ == '__main__':
    main()
