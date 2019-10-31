import math
import torch.nn as nn

# Test normal init vs. xavier normal init
# Test no dropout vs. dropout

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, relu=True):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True) if relu else nn.Identity())
    def forward(self, x):
        return self.net(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, relu=True):
        super(ConvTransposeBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True) if relu else nn.Identity())
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size,
            stride=1, padding=0, dropout=False):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(channels))
    def forward(self, x):
        return self.net(x) + x

class LeakyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, instance_norm=True, leaky_relu=True):
        super(LeakyConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels) if instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True) if leaky_relu else nn.Identity())
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, in_channels, residual_layers=9):
        super(Generator, self).__init__()
        conv_layers = []

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=1, padding=3),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1))

        # Transformer
        self.transformer = nn.Sequential(
            *[ResidualBlock(256, kernel_size=3, stride=1, padding=1,
                dropout=False) for _ in range(residual_layers)])

        # Decoder
        self.decoder = nn.Sequential(
            ConvTransposeBlock(256, 128, kernel_size=3, stride=2, padding=1),
            ConvTransposeBlock(128, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 3, kernel_size=7, stride=1, padding=3, relu=False),
            nn.Tanh())

        # Generator
        self.net = nn.Sequential(self.encoder, self.transformer, self.decoder)

        # Initialize weights
        for module in self.net.modules():
            if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        # Discriminator
        self.net = nn.Sequential(
            LeakyConvBlock(in_channels, 64, kernel_size=4, stride=2, padding=1),
            LeakyConvBlock(64, 128, kernel_size=4, stride=2, padding=1),
            LeakyConvBlock(128, 256, kernel_size=4, stride=2, padding=1),
            LeakyConvBlock(256, 512, kernel_size=4, stride=2, padding=1),
            LeakyConvBlock(512, 512, kernel_size=4, stride=1, padding=1),
            LeakyConvBlock(512, 1, kernel_size=4, stride=1, padding=1,
                instance_norm=False, leaky_relu=False))

        # Initialize weights
        for module in self.net.modules():
            if type(module) == nn.Conv2d:
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        return self.net(x)
