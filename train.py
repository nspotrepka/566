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

    # Load training data
    train_dataset = Composite(
        size=params.data_size,
        image_channels=params.image_channels,
        audio_channels=params.audio_channels,
        cache=params.cache != 0,
        validation=False
    )
    train_loader = setup.load(train_dataset, batch_size)

    # Load validation data
    val_dataset = Composite(
        size=params.data_size,
        image_channels=params.image_channels,
        audio_channels=params.audio_channels,
        cache=params.cache != 0,
        validation=True
    )
    val_loader = setup.load(val_dataset, batch_size)

    # Create model
    in_channels = train_dataset.in_channels
    out_channels = train_dataset.out_channels
    model = CycleGAN(
        train_loader=train_loader,
        val_loader=val_loader,
        in_channels=in_channels,
        out_channels=out_channels,
        g_filters=params.g_filters,
        d_filters=params.d_filters,
        residual_layers=params.residual_layers,
        extra=params.d_extra != 0,
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
        epochs=params.epochs
    )

    # Set up trainer
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints'),
        monitor='val_loss',
        verbose=True,
        prefix=params.prefix
    )
    if setup.cuda_is_available():
        trainer = Trainer(
            distributed_backend='dp',
            gpus=setup.cuda_device_count(),
            checkpoint_callback=checkpoint,
            early_stop_callback=None,
            max_nb_epochs=params.epochs
        )
    else:
        trainer = Trainer(
            checkpoint_callback=checkpoint,
            early_stop_callback=None,
            max_nb_epochs=params.epochs
        )

    # Train
    setup.init_audio()
    trainer.fit(model)
    setup.shutdown_audio()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prefix', default='', help='model checkpoint prefix')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--data_size', type=int, default=256, help='size of converted image or audio')
    parser.add_argument('--image_channels', type=int, default=3, help='number of channels to use in source image')
    parser.add_argument('--audio_channels', type=int, default=2, help='number of channels to use in source audio')
    parser.add_argument('--cache', type=int, default=1, help='cache audio/image data after loading: 0 or 1')
    parser.add_argument('--g_filters', type=int, default=64, help='number of base filters in the generator')
    parser.add_argument('--d_filters', type=int, default=64, help='number of base filters in the discriminator')
    parser.add_argument('--residual_layers', type=int, default=9, help='number of residual layers')
    parser.add_argument('--d_extra', type=int, default=0, help='use extra leaky conv layer in discriminator')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--dropout', type=int, default=0, help='use dropout in residual layers: 0 or 1')
    parser.add_argument('--init_type', default='normal', help='weight initialization type: normal, xavier, kaiming, orthogonal')
    parser.add_argument('--init_scale', type=float, default=0.02, help='weight initialization scale')
    parser.add_argument('--pool_size', type=int, default=50, help='size of data pool, which stores previous data batches')
    parser.add_argument('--lambda_a', type=float, default=10.0, help='coefficient for cycle A loss')
    parser.add_argument('--lambda_b', type=float, default=10.0, help='coefficient for cycle B loss')
    parser.add_argument('--lambda_id', type=float, default=0.0, help='coefficient for identity loss, input/output dimension must match')
    parser.add_argument('--lambda_d', type=float, default=0.5, help='coefficient for discriminator loss')

    params = parser.parse_args()
    main(params)
