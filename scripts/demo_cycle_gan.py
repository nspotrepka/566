import common.setup as setup
from data.composite import Composite
from models.cycle_gan import CycleGAN
from pytorch_lightning import Trainer
import time

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    device = setup.device()

    # Set up datasets
    # Size can be either 128 or 256
    size = 128
    image_channels = 3
    audio_channels = 2
    batch_size = 1
    train = Composite(size, image_channels, audio_channels, type='train')
    val = Composite(size, image_channels, audio_channels, type='validation')
    test = Composite(size, image_channels, audio_channels, type='test')

    # Set up loaders
    train_loader = setup.load(train, batch_size)
    val_loader = setup.load(val, batch_size)
    test_loader = setup.load(test, batch_size)

    # Set up models
    in_channels = train.in_channels
    out_channels = train.out_channels
    g_filters = 64
    d_filters = 64
    model = setup.parallel(CycleGAN(train_loader, val_loader, test_loader,
        in_channels, out_channels, g_filters, d_filters))
    model = model.to(device)

    trainer = Trainer()

    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    main()
