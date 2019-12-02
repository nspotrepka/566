from argparse import Namespace
from collections import OrderedDict
from common.loss import GANLoss
from common.pool import Pool
import itertools
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.generator import Generator
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

# Test normal init vs. xavier normal init vs. kaiming vs. orthogonal
# Test generator 64 vs. 32
# Figure out how many epochs to train (128, 256, 512)

class CycleGAN(pl.LightningModule):
    def __init__(self, train_loader, val_loader,
                 in_channels, out_channels, g_filters=64, d_filters=64,
                 residual_layers=9, dropout=False, learning_rate=0.0002,
                 beta_1=0.5, beta_2=0.999, init_type='normal', init_scale=0.02,
                 pool_size_a=50, pool_size_b=50, lambda_a=10.0, lambda_b=10.0,
                 lambda_id=0.0, lambda_g=1, lambda_d=1, epochs=200):
        super(CycleGAN, self).__init__()

        self.hparams = Namespace(**{
            'in_channels': in_channels,
            'out_channels': out_channels,
            'g_filters': g_filters,
            'd_filters': d_filters,
            'residual_layers': residual_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'init_type': init_type,
            'init_scale': init_scale,
            'pool_size_a': pool_size_a,
            'pool_size_b': pool_size_b,
            'lambda_a': lambda_a,
            'lambda_b': lambda_b,
            'lambda_id': lambda_id,
            'lambda_g': lambda_g,
            'lambda_d': lambda_d,
            'epochs': epochs
        })

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_id = lambda_id
        self.lambda_g = lambda_g
        self.lambda_d = lambda_d
        self.epochs = epochs
        self.g_loss = 0
        self.d_loss = 0
        self.g_val_loss = 0
        self.d_val_loss = 0
        self.gd = 0

        # A -> B
        self.gen_a_to_b = Generator(in_channels, out_channels, g_filters,
                                    residual_layers, dropout, init_type,
                                    init_scale)
        # B -> A
        self.gen_b_to_a = Generator(out_channels, in_channels, g_filters,
                                    residual_layers, dropout, init_type,
                                    init_scale)
        self.gen = nn.ModuleList([self.gen_a_to_b, self.gen_b_to_a])

        # Dimensions must match if using identity loss
        if lambda_id > 0.0:
            assert in_channels == out_channels

        # Data Pools
        self.fake_a_pool = Pool(pool_size_a)
        self.fake_b_pool = Pool(pool_size_b)

        # A -> real/fake
        self.dis_a = Discriminator(in_channels, d_filters, init_type,
                                   init_scale)
        # B -> real/fake
        self.dis_b = Discriminator(out_channels, d_filters, init_type,
                                   init_scale)
        self.dis = nn.ModuleList([self.dis_a, self.dis_b])

        # Loss Functions
        self.loss_func_gan = GANLoss()
        self.loss_func_cycle = nn.L1Loss()
        self.loss_func_id = nn.L1Loss()

    def forward(self, input):
        image_batch, audio_batch = input
        self.real_a, _ = image_batch
        self.real_b, _ = audio_batch
        self.fake_a = self.gen_b_to_a(self.real_b)
        self.fake_b = self.gen_a_to_b(self.real_a)
        self.cycle_a = self.gen_b_to_a(self.fake_b)
        self.cycle_b = self.gen_a_to_b(self.fake_a)

    def backward_g(self):
        if self.lambda_id > 0.0:
            self.id_a = self.gen_b_to_a(self.real_a)
            self.id_b = self.gen_a_to_b(self.real_b)
            self.loss_id_a = self.loss_func_id(self.id_a, self.real_a)
            self.loss_id_a *= self.lambda_a * self.lambda_id
            self.loss_id_b = self.loss_func_id(self.id_b, self.real_b)
            self.loss_id_b *= self.lambda_b * self.lambda_id
            self.loss_id = self.loss_id_a + self.loss_id_b
        else:
            self.loss_id_a = 0.0
            self.loss_id_b = 0.0
            self.loss_id = 0.0

        self.loss_gan_a = self.loss_func_gan(self.dis_a(self.fake_a), True)
        self.loss_gan_a *= self.lambda_g
        self.loss_gan_b = self.loss_func_gan(self.dis_b(self.fake_b), True)
        self.loss_gan_b *= self.lambda_g
        self.loss_gan = self.loss_gan_a + self.loss_gan_b

        self.loss_cycle_a = self.loss_func_cycle(self.cycle_a, self.real_a)
        self.loss_cycle_a *= self.lambda_a
        self.loss_cycle_b = self.loss_func_cycle(self.cycle_b, self.real_b)
        self.loss_cycle_b *= self.lambda_b
        self.loss_cycle = self.loss_cycle_a + self.loss_cycle_b

        self.loss_g = self.loss_gan + self.loss_cycle + self.loss_id

        losses = {
            'loss_g': self.loss_g,
            'loss_gan_a': self.loss_gan_a,
            'loss_gan_b': self.loss_gan_b,
            'loss_cycle_a': self.loss_cycle_a,
            'loss_cycle_b': self.loss_cycle_b
        }
        return self.loss_g, losses

    def backward_d_func(self, net, real, fake):
        loss_real = self.loss_func_gan(net(real), True)
        loss_fake = self.loss_func_gan(net(fake), False)
        return loss_real + loss_fake

    def backward_d(self, use_pool=True):
        device_a = next(self.dis_a.parameters()).device
        device_b = next(self.dis_b.parameters()).device
        if use_pool:
            fake_a = self.fake_a_pool.query(self.fake_a, device_a).detach()
            fake_b = self.fake_b_pool.query(self.fake_b, device_b).detach()
        else:
            fake_a = self.fake_a
            fake_b = self.fake_b
        self.loss_d_a = self.backward_d_func(self.dis_a, self.real_a, fake_a)
        self.loss_d_a *= self.lambda_d
        self.loss_d_b = self.backward_d_func(self.dis_b, self.real_b, fake_b)
        self.loss_d_b *= self.lambda_d
        self.loss_d = self.loss_d_a + self.loss_d_b

        losses = {
            'loss_d': self.loss_d,
            'loss_d_a': self.loss_d_a,
            'loss_d_b': self.loss_d_b
        }
        return self.loss_d, losses

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        # Optimizers
        self.optimizer_g = optim.Adam(
            itertools.chain(
                self.gen_a_to_b.parameters(),
                self.gen_b_to_a.parameters()),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2))
        self.optimizer_d = optim.Adam(
            itertools.chain(
                self.dis_a.parameters(),
                self.dis_b.parameters()),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2))
        self.optimizers = [self.optimizer_g, self.optimizer_d]

        def lr_lambda(epoch):
            half = self.epochs // 2
            return 1.0 - max(0, epoch - half) / float(half + 1)

        # Schedulers
        self.schedulers = [
            optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=lr_lambda),
            optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lr_lambda)
        ]

        return self.optimizers, self.schedulers

    def training_step(self, batch, batch_nb, optimizer_i):
        if self.gd < -1 or self.gd > 1:
            print('self.gd is not -1, 0, 1')
        if self.gd == 0:
            self.forward(batch)
        if optimizer_i == 0:
            # Train generator
            loss, dict = self.backward_g()
            self.gd += 1
        elif optimizer_i == 1:
            # Train discriminator
            loss, dict = self.backward_d()
            self.gd -= 1

        return OrderedDict({
            'loss': loss,
            'progress_bar': dict,
            'log': dict
        })

    def validation_step(self, batch, batch_nb):
        self.forward(batch)
        g_loss, _ = self.backward_g()
        d_loss, _ = self.backward_d()

        return OrderedDict({
            'val_loss': 0.5 * (g_loss + d_loss)
        })

    def validation_end(self, outputs):
        avg_loss = torch.stack([step['val_loss'] for step in outputs]).mean()
        dict = {
            'val_loss': avg_loss
        }
        return OrderedDict({
            'avg_val_loss': avg_loss,
            'progress_bar': dict,
            'log': dict
        })
