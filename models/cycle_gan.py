import itertools
import math
import torch
import torch.nn as nn
import torch.optim as optim

# Test normal init vs. xavier normal init
# Test no dropout vs. dropout
# Test generator 64 vs. 32

def init_module(module, init_type, init_scale):
    if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
        if init_type == 'normal':
            nn.init.normal_(module.weight, std=init_scale)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight, gain=init_scale)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=init_scale)
        else:
            raise ValueError("Unknown initialization method: [%s]" % init_type)
        nn.init.constant_(module.bias, 0)
    elif type(module) == nn.InstanceNorm2d:
        nn.init.normal_(module.weight, 1, init_scale)
        nn.init.constant_(module.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, relu=True):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(True) if relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, relu=True):
        super(ConvTransposeBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                padding=padding, output_padding=padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(True) if relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size,
            stride=1, padding=0, dropout=False):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size, stride),
            nn.InstanceNorm2d(channels, affine=True))

    def forward(self, x):
        return self.net(x) + x

class LeakyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, instance_norm=True, leaky_relu=True):
        super(LeakyConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True)
                if instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True) if leaky_relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

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

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, g_filters=64, d_filters=64,
                 residual_layers=9, dropout=False, learning_rate=0.0002,
                 beta_1=0.5, init_type='normal', init_scale=0.02,
                 lambda_a=10, lambda_b=10, lambda_id=0, training=True):
        super(CycleGAN, self).__init__()

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_id = lambda_id
        self.training = training
        self.fake_a = None
        self.fake_b = None
        self.cycle_a = None
        self.cycle_b = None

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
                assert(in_channels == out_channels)

            # Data Pools

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

            # Optimizers
            self.optimizer_g = optim.Adam(
                itertools.chain(
                    self.gen_a_to_b.parameters(),
                    self.gen_b_to_a.parameters()),
                lr=learning_rate,
                betas=(beta_1, 0.999))
            self.optimizer_d = optim.Adam(
                itertools.chain(
                    self.dis_a.parameters(),
                    self.dis_b.parameters()),
                lr=learning_rate,
                betas=(beta_1, 0.999))
            self.optimizers = [self.optimizer_g, self.optimizer_d]

            # Schedulers

    def forward(self, a, b):
        self.real_a = a
        self.real_b = b
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
        self.loss_g.backward()

    def backward_d_func(self, net, real, fake):
        loss_real = self.loss_func_gan(net(real), True)
        loss_fake = self.loss_func_gan(net(fake.detach()), False)
        return loss_real + loss_fake

    def backward_d(self):
        fake_a = self.fake_a #self.fake_a_pool.query(self.fake_a)
        fake_b = self.fake_b #self.fake_b_pool.query(self.fake_b)
        self.loss_d_a = self.backward_d_func(self.dis_a, self.real_a, fake_a)
        self.loss_d_b = self.backward_d_func(self.dis_b, self.real_b, fake_b)
        self.loss_d = self.loss_d_a + self.loss_d_b
        self.loss_d *= 0.5
        self.loss_d.backward()

    def train(self, a, b):
        self.forward(a, b)

        # Disable gradients in discriminators
        for p in self.dis.parameters():
            p.requires_grad = False

        # Optimize generator
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # Enable gradients in discriminators
        for p in self.dis.parameters():
            p.requires_grad = True

        # Optimize discriminator
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
