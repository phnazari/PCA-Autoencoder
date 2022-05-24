import torch
from torch import nn


class LinearAutoEncoder(nn.Module):
    def __init__(self, load=False, path=None, **kwargs):
        super().__init__()

        self.input_dim = kwargs["input_shape"]

        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=kwargs["input_shape"])
        )

        # if we should load existing model
        if load:
            print("[load] model...\n")

            self.load_state_dict(torch.load(path))
            self.eval()

        # register hook for latent layer
        self.encoder.register_forward_hook(get_activation(0))

    def forward(self, features):
        features = self.encoder(features)
        features = self.decoder(features)
        return features


activation = {}


def get_activation(j):
    """
    :param j: number of layer, zero based
    :return: activations at layer i
    """

    def hook(model, input, output):
        activation[j] = output.detach()

    return hook
