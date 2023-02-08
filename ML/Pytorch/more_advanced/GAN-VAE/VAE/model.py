import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma*epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma

class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2mu2 = nn.Linear(z_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        self.relu = nn.ReLU()
        self.rectify = nn.Softplus() #nn.ReLU()
    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu2(self.hid_2mu(h)), self.rectify(self.hid_2sigma(h))
        #mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def forward(self, x):
        mu, sigma = self.encode(x)
        return mu, sigma

class Generator(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()


    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, z_new):
        x_reconstructed = self.decode(z_new)
        return x_reconstructed

#if __name__ == "__main__":
    #x = torch.randn(4, 28*28)
    #vae = VariationalAutoEncoder(input_dim=784)
    #disc = Discriminator(input_dim=784)
    #gen = Generator()
    #x_reconstructed, mu, sigma = vae(x)
    #mu, sigma = disc(x)
    #x_reconstructed = gen(mu, sigma)
    #print(x_reconstructed.shape)
    #print(mu.shape)
    #print(sigma.shape)



