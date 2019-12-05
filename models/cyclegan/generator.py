from common.weights import Initializer
from models.cyclegan.layers import ConvBlock
from models.cyclegan.layers import ConvTransposeBlock
from models.cyclegan.layers import LeakyConvBlock
from models.cyclegan.layers import ResidualBlock
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, filters=64, residual_layers=9,
                 dropout=False, skip=False, init_type='normal',
                 init_scale=0.02):
        super(Generator, self).__init__()

        self.tanh = nn.Tanh()
        self.skip = skip

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
                filters * 4, filters * 2, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvTransposeBlock(
                filters * 2, filters, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvBlock(
                filters, out_channels, kernel_size=7, stride=1, padding=3,
                relu=False))

        # Generator
        self.net = nn.Sequential(self.encoder, self.transformer, self.decoder)

        # Initialize weights
        init_weights = Initializer(init_type, init_scale)
        for module in self.net.modules():
            init_weights(module)

    def forward(self, x):
        if self.skip:
            return self.tanh(self.net(x) + x)
        else:
            return self.tanh(self.net(x))

class VAEGenerator(nn.Module):
    def __init__(self, size, in_channels, out_channels, filters=64, z_size=256,
                 hidden_layers=2, hidden_size=384, init_type='xavier',
                 init_scale=0.02):
        super(VAEGenerator, self).__init__()

        assert size == 128 or size == 256 or size == 512

        self.size = size

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels, filters, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters, filters, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters, filters * 2, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters * 2, filters * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(
                filters * 2, filters * 4, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters * 4, filters * 4, kernel_size=3, stride=2, padding=1),
            ConvBlock(
                filters * 4, filters * 8, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters * 8, filters * 8, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                filters * 8, filters * 8, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(size // 16),
            nn.Flatten(),
            nn.Linear(2 * 2 * filters * 8, int(hidden_size)),
            nn.ReLU(True),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True)) for _ in range(hidden_layers)])

        self.calculate_mu = nn.Linear(hidden_size, z_size)
        self.calculate_logvar = nn.Linear(hidden_size, z_size)
        self.resize = nn.Linear(z_size, size * size // 64)

        # Decoder
        self.decoder = nn.Sequential(
            ConvTransposeBlock(
                1, filters * 8, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvTransposeBlock(
                filters * 8, filters * 4, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvTransposeBlock(
                filters * 4, filters * 2, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvTransposeBlock(
                filters * 2, filters, kernel_size=3, stride=1, padding=1),
            ConvTransposeBlock(
                filters, out_channels, kernel_size=3, stride=1, padding=1))

    def reparameterize(self, mu, logvar, training=True):
        if training:
            sigma = logvar.mul(0.5).exp_()
            return torch.normal(mu, sigma)
        else:
            return mu

    def sample(self, z):
        after = self.resize(z)
        after = after.view((-1, 1, self.size // 8, self.size // 8))
        return self.decoder(after)

    def forward(self, x):
        before = self.encoder(x)
        mu = self.calculate_mu(before)
        logvar = self.calculate_logvar(before)
        z = self.reparameterize(mu, logvar)
        return self.sample(z), z, mu, logvar
