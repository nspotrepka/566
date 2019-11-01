import common.setup as setup
# from data.gaped import GAPED
# from data.pmemo import PMEmo
from data.composite import Composite
from models.cycle_gan import CycleGAN
from pytorch_lightning import Trainer
import time

def main():
    print('PyTorch', setup.torch_version())
    print('CUDA is available:', setup.cuda_is_available())
    print('CUDA device count:', setup.cuda_device_count())

    device = setup.device()

    # size can be either 128 or 256
    data = Composite(128, image_channels=3, audio_channels=2, type='train')
    loader = setup.load(data, batch_size=1)

    in_channels = data.in_channels
    out_channels = data.out_channels
    model = setup.parallel(CycleGAN(loader, in_channels, out_channels, 32, 64))
    model = model.to(device)

    trainer = Trainer()

    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    main()
