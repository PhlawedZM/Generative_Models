import torch
import util.VAEUtil as util
import numpy as np
from torchsummary import summary
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(1024),
        )

        self.fc = None

        self.flatten = nn.Flatten(1)

        self.mean = None
        self.variance = None
        self.size = None
        self.shape = None

    def forward(self, x):
        x = self.encode(x)
        self.shape = x.shape

        x = self.flatten(x)

        self.size = util.numel(x)
        self.fc = nn.Linear(self.size, self.latent_dim).to(x.device)

        self.mean = self.fc(x)
        self.variance = self.fc(x)

        return self.mean, self.variance, self.size, self.shape


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, space_size=65536, shape=(-1, 1024, 8, 8)):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.space_size = space_size
        self.shape = shape

        self.fc = nn.Linear(latent_dim, space_size)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view((-1, *self.shape[1:]))
        x = self.decode(x)

        return x


def test():
    latent_dim = 256
    width = 512
    height = 512
    encoder = Encoder(latent_dim=latent_dim).cuda()
    x = torch.randn(1, 3, width, height).cuda()
    summary(encoder, (3, width, height))
    mean, var, size, shape = encoder(x)

    decoder = Decoder(latent_dim=latent_dim, space_size=size, shape=shape).cuda()

    epsilon = torch.randn_like(mean).cuda()
    std = torch.exp(0.5 * var).cuda()
    z = mean + std * epsilon
    img = decoder(z)

    summary(decoder, (latent_dim,))


test()
