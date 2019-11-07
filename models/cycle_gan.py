from common.layers import init_module
from common.layers import ConvBlock
from common.layers import ConvTransposeBlock
from common.layers import LeakyConvBlock
from common.layers import ResidualBlock
from common.pool import Pool
import itertools
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

# Test normal init vs. xavier normal init
# Test no dropout vs. dropout
# Test generator 64 vs. 32
# Test input size 128 vs 256 (vs 512?)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, filters=64, residual_layers=9,
                 dropout=False, init_type='normal', init_scale=0.02):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels, filters, kernel_size=7, stride=1, padding=3),
            ConvBlock(
                filters, filters * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(
                filters * 2, filters * 4, kernel_size=3, stride=2, padding=1))

        # Transformer
        self.transformer = nn.Sequential(
            *[ResidualBlock(filters * 4, kernel_size=3, stride=1, padding=1,
                dropout=dropout) for _ in range(residual_layers)])

        # Decoder
        self.decoder = nn.Sequential(
            ConvTransposeBlock(
                filters * 4, filters * 2, kernel_size=3, stride=2, padding=1),
            ConvTransposeBlock(
                filters * 2, filters, kernel_size=3, stride=2, padding=1),
            ConvBlock(
                filters, out_channels, kernel_size=7, stride=1, padding=3,
                relu=False),
            nn.Tanh())

        # Generator
        self.net = nn.Sequential(self.encoder, self.transformer, self.decoder)

        # Initialize weights
        for module in self.net.modules():
            init_module(module, init_type, init_scale)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, filters=64, init_type='normal',
                 init_scale=0.02):
        super(Discriminator, self).__init__()

        # Discriminator
        p = (1, 2, 1, 2)
        self.net = nn.Sequential(
            LeakyConvBlock(
                in_channels, filters, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters, filters * 2, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters * 2, filters * 4, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters * 4, filters * 8, kernel_size=4, stride=1, padding=p),
            LeakyConvBlock(
                filters * 8, 1, kernel_size=4, stride=1, padding=p,
                instance_norm=False, leaky_relu=False))

        # Initialize weights
        for module in self.net.modules():
            init_module(module, init_type, init_scale)

    def forward(self, x):
        return self.net(x)

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss_func = nn.MSELoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target = self.real_label
        else:
            target = self.fake_label
        target = target.expand_as(prediction)
        return self.loss_func(prediction, target)

class CycleGAN(pl.LightningModule):
    def __init__(self,
                 loader, in_channels, out_channels, g_filters=64, d_filters=64,
                 residual_layers=9, dropout=False, learning_rate=0.0002,
                 beta_1=0.5, beta_2=0.999, init_type='normal', init_scale=0.02,
                 pool_size=0, lambda_a=10.0, lambda_b=10.0, lambda_id=0.0,
                 n_flat=100, n_decay=100, training=True):
        super(CycleGAN, self).__init__()

        self.loader = loader
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_id = lambda_id
        self.n_flat = n_flat
        self.n_decay = n_decay
        self.training = training

        # A -> B
        self.gen_a_to_b = Generator(in_channels, out_channels, g_filters,
                                    residual_layers, dropout, init_type,
                                    init_scale)
        # B -> A
        self.gen_b_to_a = Generator(out_channels, in_channels, g_filters,
                                    residual_layers, dropout, init_type,
                                    init_scale)
        self.gen = nn.ModuleList([self.gen_a_to_b, self.gen_b_to_a])

        if training:
            # Dimensions must match if using identity loss
            if lambda_id > 0.0:
                assert in_channels == out_channels

            # Data Pools
            self.fake_a_pool = Pool(pool_size)
            self.fake_b_pool = Pool(pool_size)

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
        self.loss_gan_b = self.loss_func_gan(self.dis_b(self.fake_b), True)
        self.loss_gan = self.loss_gan_a + self.loss_gan_b

        self.loss_cycle_a = self.loss_func_cycle(self.cycle_a, self.real_a)
        self.loss_cycle_a *= self.lambda_a
        self.loss_cycle_b = self.loss_func_cycle(self.cycle_b, self.real_b)
        self.loss_cycle_b *= self.lambda_b
        self.loss_cycle = self.loss_cycle_a + self.loss_cycle_b

        self.loss_g = self.loss_gan + self.loss_cycle + self.loss_id
        return self.loss_g

    def backward_d_func(self, net, real, fake):
        loss_real = self.loss_func_gan(net(real), True)
        loss_fake = self.loss_func_gan(net(fake.detach()), False)
        return loss_real + loss_fake

    def backward_d(self):
        fake_a = self.fake_a_pool.query(self.fake_a)
        fake_b = self.fake_b_pool.query(self.fake_b)
        self.loss_d_a = self.backward_d_func(self.dis_a, self.real_a, fake_a)
        self.loss_d_b = self.backward_d_func(self.dis_b, self.real_b, fake_b)
        self.loss_d = self.loss_d_a + self.loss_d_b
        self.loss_d *= 0.5
        return self.loss_d

    @pl.data_loader
    def train_dataloader(self):
        return self.loader

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
            return 1.0 - max(0, epoch - self.n_flat) / float(self.n_decay + 1)

        # Schedulers
        self.schedulers = [
            optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=lr_lambda),
            optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lr_lambda)
        ]

        return self.optimizers, self.schedulers

    def training_step(self, batch, batch_nb, optimizer_i):
        if optimizer_i == 0:
            # Train generator
            self.forward(batch)
            loss = self.backward_g()
            dict = {'g_loss': loss}
        elif optimizer_i == 1:
            # Train discriminator
            loss = self.backward_d()
            dict = {'d_loss': loss}

        return {
            'loss': loss,
            'progress_bar': dict,
            'log': dict
        }
