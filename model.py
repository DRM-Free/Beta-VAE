"""model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# Auxiliary network takes a pair of encoded images as input and tries to determine the pairwise ambiguity or similarity between those images


class Auxiliary_network(nn.Module):
    def __init__(self, z_dim):
        super(Auxiliary_network, self).__init__()
        self.z_dim = z_dim
        size = 5  # previously 20

        # This is a classifier (ambiguous/not ambiguous) so the activation functions must be steep
        # self.net = nn.Sequential(
        #     nn.Linear(2*z_dim, size),
        #     torch.nn.Prelu(),
        #     nn.Dropout(0.1),
        #     nn.Linear(size, size),
        #     torch.nn.Prelu(),
        #     nn.Dropout(0.1),
        #     nn.Linear(size, size),
        #     torch.nn.Prelu(),
        #     nn.Linear(size, 1),
        # )

        # self.net = nn.Sequential(
        #     nn.Linear(2*z_dim, 10),
        #     torch.nn.LeakyReLU(),
        #     nn.Linear(10, 10),
        #     torch.nn.LeakyReLU(),
        #     nn.Linear(10, 1),
        #     nn.Hardtanh()
        #     # nn.LogSigmoid()
        # )

        self.net = nn.Sequential(
            nn.Linear(2*z_dim, size),
            torch.nn.Sigmoid(),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            torch.nn.PReLU(),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            torch.nn.PReLU(),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            torch.nn.PReLU(),
            nn.Linear(size, 1),
            nn.Sigmoid()
            # torch.nn.PReLU()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        return self.net(x)


class Position_auxiliary_encoder(nn.Module):
    def __init__(self, pos_dim, code_dim):
        super(Position_auxiliary_encoder, self).__init__()
        self.pos_dim = pos_dim
        self.code_dim = code_dim
        size = 20
        # This is not a classifier but a function estimator, so the activation functions should not be too steep
        self.net = nn.Sequential(
            nn.Linear(self.pos_dim, size),
            nn.Dropout(0.2),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            nn.Linear(size, size),
            nn.Dropout(0.2),
            nn.Linear(size, self.code_dim),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        return self.net(x)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            # conv2d "groups" option for grouped convolution
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    # import ant to know
    # numpy image: H x W x C
    # torch image: C X H X W
    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.PReLU(),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.PReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.PReLU(),
            nn.Linear(256, 256),  # B, 256
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.PReLU(),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.PReLU(),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
