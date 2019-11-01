import common.setup as setup
from data.gaped import GAPED
from data.pmemo import PMEmo
from models.cycle_gan import CycleGAN
import time

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    device = setup.device()

    image_data = GAPED(256)
    audio_data = PMEmo(256)
    image_loader = setup.load(image_data, batch_size=1)
    audio_loader = setup.load(audio_data, batch_size=1)

    in_channels = 3
    out_channels = 3
    model = setup.parallel(CycleGAN(in_channels, out_channels, 64, 64))
    model = model.to(device)

    count = 0
    for image_batch, audio_batch in zip(image_loader, audio_loader):
        start_time = time.time()

        image, image_emotion = image_batch
        audio, audio_emotion = audio_batch

        image = image.to(device)
        audio = audio.to(device)
        image_emotion = image_emotion.to(device)
        audio_emotion = audio_emotion.to(device)

        model.train(image, audio)

        end_time = time.time()

        count += 1
        print('Trained', count, '/', dataset.__len__())
        print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
