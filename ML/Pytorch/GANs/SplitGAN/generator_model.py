"""
Generator model for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

class Encoder(nn.Module):
    def __init__(self, img_channels, num_features=16, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.img_2hid = nn.Linear(num_features, 50)
        self.hid_2mu = nn.Linear(50, 75)
        self.hid_2mu2 = nn.Linear(75, 125)
        self.hid_2mu2_leak = nn.LeakyReLU(0.8)
        self.hid_2mu3 = nn.Linear(125, 150)
        self.hid_2mu3_leak = nn.LeakyReLU(0.75)
        self.hid_2mu4 = nn.Linear(150, 250)
        self.hid_2mu4_leak = nn.LeakyReLU(0.75)
        self.hid_2mu5 = nn.Linear(250, 375)
        self.hid_2mu5_leak = nn.LeakyReLU(0.75)
        self.hid_2mu6 = nn.Linear(375, 750)
        self.hid_2mu6_leak = nn.LeakyReLU(0.8)
        self.hid_2sigma = nn.Linear(50, 75)
        self.hid_2sigma2 = nn.Linear(75, 125)
        self.hid_2sigma2_leak = nn.LeakyReLU(0.8)
        self.hid_2sigma3 = nn.Linear(125, 150)
        self.hid_2sigma3_leak = nn.LeakyReLU(0.75)
        self.hid_2sigma4 = nn.Linear(150, 250)
        self.hid_2sigma4_leak = nn.LeakyReLU(0.75)
        self.hid_2sigma5 = nn.Linear(250, 375)
        self.hid_2sigma5_leak = nn.LeakyReLU(0.75)
        self.hid_2sigma6 = nn.Linear(375, 750)
        self.relu = nn.ReLU()
        self.rectify = nn.Softplus() #nn.ReLU()
    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu2, sigma2 = self.hid_2mu2(self.hid_2mu(h)), self.hid_2sigma2(self.hid_2sigma(h))
        mu2, sigma2 = self.hid_2mu2_leak(mu2), self.hid_2sigma2_leak(sigma2)
        mu3, sigma3 = self.hid_2mu3(mu2), self.hid_2sigma3(sigma2) #self.rectify(self.hid_2sigma(h))
        mu3, sigma3 = self.hid_2mu3_leak(mu3), self.hid_2sigma3_leak(sigma3)
        mu4, sigma4 = self.hid_2mu4(mu3), self.hid_2sigma4(sigma3)
        mu4, sigma4 = self.hid_2mu4_leak(mu4), self.hid_2sigma4_leak(sigma4)
        mu5, sigma5 = self.hid_2mu5(mu4), self.hid_2sigma5(sigma4)
        mu5, sigma5 = self.hid_2mu5_leak(mu5), self.hid_2sigma5_leak(sigma5)
        mu6, sigma6 = self.hid_2mu6(mu5), self.hid_2sigma6(sigma5)
        mu, sigma = self.hid_2mu6_leak(mu6), self.rectify(sigma6)
        #mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        #print(x.shape)
        mu, sigma = self.encode(x)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, img_channels, num_features=16, num_residuals=9):
        super().__init__()
        self.z_2hid = nn.Linear(750, 375)
        self.z_2leak = nn.LeakyReLU(0.99)
        self.z_3hid = nn.Linear(375, 250)
        self.z_3leak = nn.LeakyReLU(0.9)
        self.z_4hid = nn.Linear(250, 150)
        self.z_4leak = nn.LeakyReLU(0.9)
        self.z_5hid = nn.Linear(150, 125)
        self.z_5leak = nn.LeakyReLU(0.9)
        self.z_6hid = nn.Linear(125, 75)
        self.z_6leak = nn.LeakyReLU(0.9)
        self.z_7hid = nn.Linear(75, 50)
        self.z_7leak = nn.LeakyReLU(0.99)
        self.hid_2img = nn.Linear(50, num_features)

        #self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, z):
        h = self.z_2hid(z)
        h = self.z_2leak(h)
        x = self.z_3hid(h)
        x = self.z_3leak(x)
        x = self.z_4hid(x)
        x = self.z_4leak(x)
        x = self.z_5hid(x)
        x = self.z_5leak(x)
        x = self.z_6hid(x)
        x = self.z_6leak(x)
        x = self.z_7hid(x)
        x = self.z_7leak(x)
        x = self.hid_2img(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def test():
    img_channels = 3
    img_size = 64
    x = torch.randn((2, img_channels, img_size, img_size))
    enc = Encoder(img_channels, 16, 9)
    mu, sigma = enc(x)
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    dec = Decoder(img_channels, 16, 9) 
    #gen = Generator(img_channels, 9)
    print(dec(z).shape)


if __name__ == "__main__":
    test()
