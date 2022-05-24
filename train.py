from decimal import Decimal

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim, nn

from losses import pca_loss
from models import activation

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root="/export/ial-nfs/user/pnazari/data",
                                           train=True,
                                           transform=transform,
                                           download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,
                                           pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, model_path, epochs=50, weight_decay=0, save=True):
    # initialize model
    # create optimizer object
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # use mean-square-error as loss

    total_losses = []
    mse_losses = []
    weight_losses = []

    for epoch in range(epochs):
        batch_total_loss = 0
        batch_mse_loss = 0
        batch_weight_loss = 0

        print(f"epoch {epoch + 1} of {epochs}")

        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, model.input_dim).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss. This is where the learning of identity happens
            total_loss, mse_loss, weight_loss = pca_loss(model, outputs, batch_features, weight_decay=weight_decay)

            # compute accumulated gradients
            total_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            batch_total_loss += total_loss.item()
            batch_mse_loss += mse_loss.item()
            batch_weight_loss += weight_loss.item()
            # add the mini-batch training loss to epoch loss

        total_losses.append(batch_total_loss)
        mse_losses.append(batch_mse_loss)
        weight_losses.append(batch_weight_loss)

    if save:
        # save trained model to file
        torch.save(model.state_dict(), model_path)

    """
    Plotting
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f"PCA AE loss")

    ax1.plot(total_losses)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(r"Total loss")

    ax2.plot(mse_losses)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title("MSE loss")

    ax3.plot(weight_losses)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("loss")
    ax3.set_title("Weight Loss")

    plt.savefig(f"/export/home/pnazari/workspace/AutoEncoderVisualization/AEPCA/output/loss/loss_wd={Decimal(weight_decay):.4e}.png")
    plt.show()
