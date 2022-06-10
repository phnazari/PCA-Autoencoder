import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wd_loss(model, outputs, features, weight_decay):
    """
    Custom loss that implements weight-decay
    :param model: the model under consideration
    :param outputs: the outputs of the forward pass
    :param features: the inputs to the forward pass
    :param weight_decay: the weight-decay-rate
    :return: triple containing the total loss, the mse loss and the weight-decay loss
    """
    crit = nn.MSELoss()
    base_loss = crit(outputs, features)

    E = model.encoder[0].weight
    D = model.decoder[0].weight

    weight_loss = weight_decay / 2 * (torch.linalg.norm(E) ** 2 + torch.linalg.norm(D) ** 2)

    total_loss = base_loss + weight_loss

    return total_loss, base_loss, weight_loss
