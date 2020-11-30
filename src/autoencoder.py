from .nn_utils import *
from .loss import ae_loss, nll_loss, l2_loss, bce_loss, interp_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder:
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dim=[128, ],
            cont_idxs=[],
            bin_idxs=[],
            cat_idxs=[],
            cat_dims=[],
            cat_choice_function=F.log_softmax,
            cat_loss=nll_loss
    ):
        if type(hidden_dim) != list:
            hidden_dim = [hidden_dim, ]

        self.input_dim = int(input_dim)
        self.embeddings = EmbeddingGenerator(input_dim, cat_dims, cat_idxs)
        post_embed_dim = self.embeddings.post_embed_dim
        hidden_dim = [post_embed_dim] + hidden_dim

        self.latent_dim = int(latent_dim)
        self.hidden_dim = list(map(int, hidden_dim))
        self.cont_idxs = cont_idxs
        self.bin_idxs = bin_idxs
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cont_idxs = sorted(cont_idxs)
        self.bin_idxs = sorted(bin_idxs)
        self.cat_idxs, self.cat_dims = [], []
        if len(cat_idxs) and len(cat_dims):
            self.cat_idxs, self.cat_dims = zip(*sorted(zip(cat_idxs, cat_dims)))

        self.cat_loss = cat_loss
        self.cat_choice_function = cat_choice_function

        # Encoder
        self.encoder = [self.embeddings]
        for i in range(1, len(hidden_dim)):
            self.encoder.extend(
                    (
                        nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                        GBN(hidden_dim[i]),
                        nn.PReLU(hidden_dim[i])
                    )
            )
        self.encoder.append(nn.Linear(hidden_dim[-1], latent_dim))
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        hidden_dim = hidden_dim + [latent_dim]
        self.decoder = []
        for i in range(len(hidden_dim) - 1, 1, -1):
            self.decoder.extend(
                (
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    GBN(hidden_dim[i-1]),
                    nn.PReLU(hidden_dim[i-1])
                )
            )
        self.decoder = torch.nn.Sequential(*self.decoder)

        if len(cont_idxs) != 0:
            self.cont_net = nn.Sequential(
                    nn.Linear(hidden_dim[1], len(cont_idxs)),
                    nn.Tanh()
                    )

        if len(cat_idxs) != 0:
            self.cat_nets = nn.ModuleList()
            for i, n_cats in zip(cat_idxs, cat_dims):
                self.cat_nets.append(nn.Sequential(
                    nn.Linear(hidden_dim[1], n_cats),
                    Lambda(cat_choice_function)
                    ))

        if len(bin_idxs) != 0:
            self.bin_net = nn.Sequential(
                    nn.Linear(hidden_dim[1], len(bin_idxs)),
                    nn.Sigmoid()
                    )
        self.apply(weight_init)

    def decode(self, z):
        z = self.decoder(z)
        x_hat = []
        if hasattr(self, 'cont_net'):
            x_hat.append(self.cont_net(z))
        if hasattr(self, 'bin_net'):
            x_hat.append(self.bin_net(z))
        if hasattr(self, 'cat_nets'):
            for m in self.cat_nets:
                x_hat.append(m(z))
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def decode_sample(self, z):
        x_hat = self.decode(z)
        x_cont, x_bin, x_cat = [], [], []
        if hasattr(self, 'cont_net'):
            x_cont = x_hat.pop(0)
        if hasattr(self, 'bin_net'):
            x_bin = x_hat.pop(0)
        if hasattr(self, 'cat_nets'):
            for _ in self.cat_idxs:
                x_cat.append(torch.argmax(x_hat.pop(0), dim=1))
        x = []
        cont_c, bin_c, cat_c = 0, 0, 0
        for i in range(self.input_dim):
            if i in self.cont_idxs:
                x.append(x_cont[:, cont_c].reshape(-1, 1))
                cont_c += 1
            elif i in self.bin_idxs:
                x.append(x_bin[:, bin_c].reshape(-1, 1) > 0.5)
                bin_c += 1
            elif i in self.cat_idxs:
                x.append(x_cat[cat_c].reshape(-1, 1))
                cat_c += 1
        x = torch.cat(x, dim=1)
        return x

    def loss(self, output, target, **kwargs):
        loss = {'mse': 0.0, 'nll': 0.0, 'bce': 0.0, 'reg': 0.0, 'interp_reg': 0.0}
        x_hat = output[0]
        z = output[1]
        l2_weight = kwargs.get('l2_weight', 0.0)
        interp_weight = kwargs.get('interp_weight', 0.0)
        cont_weight = len(self.cont_idxs) / (len(self.cont_idxs) + len(self.bin_idxs) + len(self.cat_idxs))
        cat_weight = len(self.cat_idxs) / (len(self.cont_idxs) + len(self.bin_idxs) + len(self.cat_idxs))
        bin_weight = len(self.bin_idxs) / (len(self.cont_idxs) + len(self.bin_idxs) + len(self.cat_idxs))
        if len(self.cont_idxs):
            loss['mse'] += cont_weight * ae_loss(x_hat, target[:, self.cont_idxs], **kwargs)
        if len(self.bin_idxs):
            out = x_hat.pop(0)
            loss['bce'] += bin_weight * bce_loss(out, target[:, self.bin_idxs])
        if len(self.cat_idxs):
            for idx in self.cat_idxs:
                out = x_hat.pop(0)
                loss['nll'] += cat_weight * self.cat_loss(out, target[:, idx].long())
        if l2_weight:
            loss['latent_reg'] += l2_weight * l2_loss(z)
        if interp_weight:
            loss['interp_reg'] += interp_weight * interp_loss(z, kwargs['labels'])
        return loss
