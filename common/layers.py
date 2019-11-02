import torch.nn as nn

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
