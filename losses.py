import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = 1e-4
beta = .5


def pca_loss(model, outputs, features, weight_decay):
    crit = nn.MSELoss()
    base_loss = crit(outputs, features)

    E = model.encoder[0].weight
    D = model.decoder[0].weight

    weight_loss = weight_decay/2 * (torch.linalg.norm(E)**2 + torch.linalg.norm(D)**2)

    total_loss = base_loss + weight_loss

    """
    custom_loss_1 = 0
    custom_loss_1 = - alpha * torch.linalg.norm(E[0] @ features.T) / torch.linalg.norm(E[0])

    custom_loss_2 = 0
    custom_loss_2 = - alpha * torch.linalg.norm(E[1] @ features.T) / torch.linalg.norm(E[1])
    """

    # TODO: implement the second principal component

    return total_loss, base_loss, weight_loss
