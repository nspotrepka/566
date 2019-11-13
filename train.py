from argparse import ArgumentParser
import common.setup as setup
from data.composite import Composite
from models.cyclegan.model import CycleGAN
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.pt_callbacks import ModelCheckpoint

def main(params):
    # This is an unsafe, unsupported, undocumented workaround
    setup.allow_kmp_duplicate_lib()

    # Size can be 128, 256, or 512
    size = params.data_size
    image_channels = params.image_channels
    audio_channels = params.audio_channels
    batch_size = params.batch_size

    # Load data
    dataset = Composite(
        size=params.data_size,
        image_channels=params.image_channels,
        audio_channels=params.audio_channels,
        cache=params.cache != 0
    )
    loader = setup.load(dataset, batch_size)

    # Create model
    in_channels = dataset.in_channels
    out_channels = dataset.out_channels
    model = CycleGAN(
        loader=loader,
        in_channels=in_channels,
        out_channels=out_channels,
        g_filters=params.g_filters,
        d_filters=params.d_filters,
        residual_layers=params.residual_layers,
        dropout=params.dropout != 0,
        learning_rate=params.lr,
        beta_1=params.b1,
        beta_2=params.b2,
        init_type=params.init_type,
        init_scale=params.init_scale,
        pool_size=params.pool_size,
        lambda_a=params.lambda_a,
        lambda_b=params.lambda_b,
        lambda_id=params.lambda_id,
        n_flat=params.epochs // 2,
        n_decay=params.epochs // 2
    )

    # Set up trainer
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
            max_nb_epochs=params.epochs
        )
    else:
        trainer = Trainer(
            checkpoint_callback=checkpoint,
            max_nb_epochs=params.epochs
        )

    # Train
    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--data_size', type=int, default=128, help='size of converted audio/image')
    parser.add_argument('--audio_channels', type=int, default=2, help='number of channels to use in source audio')
    parser.add_argument('--image_channels', type=int, default=3, help='number of channels to use in source image')
    parser.add_argument('--cache', type=int, default=1, help='cache audio/image data after loading, 0 or 1')
    parser.add_argument('--g_filters', type=int, default=64, help='number of base filters in the generator')
    parser.add_argument('--d_filters', type=int, default=64, help='number of base filters in the discriminator')
    parser.add_argument('--residual_layers', type=int, default=9, help='number of residual layers')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--dropout', type=int, default=0, help='use dropout in residual layers, 0 or 1')
    parser.add_argument('--init_type', default='normal', help='weight initialization type: normal, xavier, kaiming, orthogonal')
    parser.add_argument('--init_scale', type=float, default=0.02, help='weight initialization scale')
    parser.add_argument('--pool_size', type=int, default=0, help='size of data pool, stores previous data batches')
    parser.add_argument('--lambda_a', type=float, default=10.0, help='scale in loss for cycle A')
    parser.add_argument('--lambda_b', type=float, default=10.0, help='scale in loss for cycle B')
    parser.add_argument('--lambda_id', type=float, default=0.0, help='scale in loss for identity, in/out channels must match')

    params = parser.parse_args()
    main(params)