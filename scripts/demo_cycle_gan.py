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

    data = Composite(256)
    loader = setup.load(data, batch_size=1)

    # image_data = GAPED(256)
    # image_loader = setup.load(image_data, batch_size=1)
    # audio_data = PMEmo(256)
    # audio_loader = setup.load(audio_data, batch_size=1)
    # loader = zip(image_loader, audio_loader)

    in_channels = 3
    out_channels = 4
    model = setup.parallel(CycleGAN(loader, in_channels, out_channels, 32, 64))
    model = model.to(device)

    trainer = Trainer()

    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    main()
