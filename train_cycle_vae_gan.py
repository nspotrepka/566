from argparse import ArgumentParser
import common.setup as setup
from data.composite import CompositePositive
from data.composite import CompositeNegative
from models.cyclegan.model import CycleVAEGAN
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.pt_callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset

def main(params):
    # This is an unsafe, unsupported, undocumented workaround
    setup.allow_kmp_duplicate_lib()

    dataset_args = {
        'size': params.data_size,
        'image_channels': params.image_channels,
        'audio_channels': params.audio_channels,
        'cache': params.cache != 0,
        'midi': params.midi != 0
    }

    # Load training data
    train_positive = CompositePositive(
        **dataset_args,
        validation=False
    )
    train_negative = CompositeNegative(
        **dataset_args,
        validation=False
    )
    train_dataset = ConcatDataset([train_positive, train_negative])
    train_loader = setup.load(train_dataset, params.batch_size)

    # Load validation data
    val_positive = CompositePositive(
        **dataset_args,
        validation=True
    )
    val_negative = CompositeNegative(
        **dataset_args,
        validation=True
    )
    val_dataset = ConcatDataset([val_positive, val_negative])
    val_loader = setup.load(val_dataset, params.batch_size)

    # Create model
    in_channels = train_dataset.datasets[0].in_channels
    out_channels = train_dataset.datasets[0].out_channels
    model = CycleVAEGAN(
        train_loader=train_loader,
        val_loader=val_loader,
        data_size=params.data_size,
        in_channels=in_channels,
        out_channels=out_channels,
        g_filters=params.g_filters,
        d_filters=params.d_filters,
        z_size=params.z_size,
        hidden_layers=params.hidden_layers,
        hidden_size=params.hidden_size,
        learning_rate=params.lr,
        beta_1=params.b1,
        beta_2=params.b2,
        init_type=params.init_type,
        init_scale=params.init_scale,
        pool_size_a=params.pool_size_a,
        pool_size_b=params.pool_size_b,
        lambda_a=params.lambda_a,
        lambda_b=params.lambda_b,
        lambda_id=params.lambda_id,
        lambda_g=params.lambda_g,
        lambda_d=params.lambda_d,
        lambda_kl=params.lambda_kl,
        lambda_recon=params.lambda_recon
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
    parser.add_argument('--prefix', default='', help='model checkpoint file name prefix')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--data_size', type=int, default=128, help='size of converted image or audio: 128 or 256 or 512')
    parser.add_argument('--image_channels', type=int, default=3, help='number of channels to use in source image')
    parser.add_argument('--audio_channels', type=int, default=2, help='number of channels to use in source audio')
    parser.add_argument('--cache', type=int, default=1, help='cache audio/image data after loading: 0 or 1')
    parser.add_argument('--g_filters', type=int, default=64, help='number of base filters in the generator')
    parser.add_argument('--d_filters', type=int, default=64, help='number of base filters in the discriminator')
    parser.add_argument('--z_size', type=int, default=256, help='size of encoding')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of shared layers')
    parser.add_argument('--hidden_size', type=int, default=360, help='size of hidden layers')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--init_type', default='xavier', help='weight initialization type: normal, xavier, kaiming, orthogonal')
    parser.add_argument('--init_scale', type=float, default=0.02, help='weight initialization scale')
    parser.add_argument('--pool_size_a', type=int, default=50, help='size of data pool A, which stores previous data batches')
    parser.add_argument('--pool_size_b', type=int, default=50, help='size of data pool B, which stores previous data batches')
    parser.add_argument('--lambda_a', type=float, default=10.0, help='coefficient for cycle A loss')
    parser.add_argument('--lambda_b', type=float, default=10.0, help='coefficient for cycle B loss')
    parser.add_argument('--lambda_id', type=float, default=0.0, help='coefficient for identity loss, input/output dimension must match')
    parser.add_argument('--lambda_g', type=float, default=1.0, help='coefficient for generator loss')
    parser.add_argument('--lambda_d', type=float, default=1.0, help='coefficient for discriminator loss')
    parser.add_argument('--lambda_kl', type=float, default=1.0, help='coefficient for KL divergence loss')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='coefficient for reconstruction loss')
    parser.add_argument('--midi', type=int, default=0, help='use Lakh clean MIDI dataset instead of PMEmo: 0 or 1')

    params = parser.parse_args()
    main(params)
