import common.setup as setup
from data.composite import Composite
from models.cyclegan.model import CycleGAN
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.pt_callbacks import ModelCheckpoint

def main():
    # This is an unsafe, unsupported, undocumented workaround
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Size can be 128, 256, or 512
    size = 128
    image_channels = 3
    audio_channels = 2
    batch_size = 1

    # Load data
    dataset = Composite(size, image_channels, audio_channels, cache=True)
    loader = setup.load(dataset, batch_size)

    # Create model
    in_channels = dataset.in_channels
    out_channels = dataset.out_channels
    g_filters = 64
    d_filters = 64
    model = CycleGAN(loader, in_channels, out_channels, g_filters, d_filters)

    # Set up trainer
    epochs = 200
    checkpoint = ModelCheckpoint(
        filepath=os.getcwd(),
        verbose=True,
        save_best_only=False,
        save_weights_only=False,
        period=1
    )
    if setup.cuda_is_available():
        gpus = range(setup.cuda_device_count())
        trainer = Trainer(
            distributed_backend='dp',
            gpus=gpus,
            checkpoint_callback=checkpoint,
            max_nb_epochs=epochs
        )
    else:
        trainer = Trainer(
            checkpoint_callback=checkpoint,
            max_nb_epochs=epochs
        )

    # Train
    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    main()
