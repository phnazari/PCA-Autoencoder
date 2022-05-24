import pickle
from decimal import Decimal

import torch
from matplotlib import pyplot as plt
from procrustes import generic

from models import activation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, pca_model, weight_decay, save=True):
    PCA_features = []
    AE_features = []
    labels = []
    AE_distances = []

    wd_AE_features = []
    wd_AE_distances = []

    for batch_features, batch_labels in test_loader:
        X = batch_features.view(-1, 784)
        X = X - X.mean(axis=0)

        print("[Calc] PCA")
        # PCA stuff
        U, S, Vh = torch.linalg.svd(X)

        batch_PCA_features = X @ Vh.T[:, :2]

        PCA_features.append(batch_PCA_features)
        labels.append(batch_labels)

        print("[Calc] AE")
        X = X.to(device)

        # AE stuff
        model(X)
        AE_batch_features = activation[0]
        AE_features.append(AE_batch_features)
        print("[Calc] Custom AE")
        pca_model(X)
        wd_AE_batch_features = activation[0]

        t = torch.tensor(generic(wd_AE_batch_features.cpu().numpy(), batch_PCA_features.cpu().numpy()).t)

        wd_AE_batch_features_transformed = wd_AE_batch_features.cpu() @ t

        wd_AE_batch_distance = (wd_AE_batch_features_transformed - batch_PCA_features).pow(2).sum(1).sqrt()
        AE_batch_distance = (batch_PCA_features - AE_batch_features.cpu()).pow(2).sum(1).sqrt()

        wd_AE_features.append(wd_AE_batch_features_transformed)
        wd_AE_distances.append(wd_AE_batch_distance)
        AE_distances.append(AE_batch_distance)

    AE_features = torch.cat(AE_features, dim=0)
    PCA_features = torch.cat(PCA_features, dim=0)
    labels = torch.cat(labels, dim=0)
    AE_distances = torch.cat(AE_distances, dim=0)
    wd_AE_features = torch.cat(wd_AE_features, dim=0)
    wd_AE_distances = torch.cat(wd_AE_distances, dim=0)

    mean_AE_distances = torch.mean(AE_distances)
    std_AE_distances = torch.std(AE_distances)
    mean_wd_AE_distances = torch.mean(wd_AE_distances)
    std_wd_AE_distances = torch.std(wd_AE_distances)

    if save:
        # save distances to file
        with open(f"/export/home/pnazari/workspace/AutoEncoderVisualization/AEPCA/output/data/distances.pkl", 'rb') as f:
            try:
                distances_dict = pickle.load(f)
            except EOFError:
                distances_dict = {"wd": [], "mean": [], "std": []}

        with open(f"/export/home/pnazari/workspace/AutoEncoderVisualization/AEPCA/output/data/distances.pkl", 'wb') as f:
            distances_dict["wd"].append(f"{Decimal(weight_decay):.4e}")
            distances_dict["mean"].append(mean_wd_AE_distances.item())
            distances_dict["std"].append(std_wd_AE_distances.item())
            pickle.dump(distances_dict, f)

    """
    Plotting
    """

    sc_kwargs = {
        "c": labels,
        "marker": ".",
        "cmap": "tab10",
        "alpha": .4,
        "s": 10,
        "edgecolors": None
    }

    PCA_features = PCA_features.detach()
    AE_features = AE_features.detach()
    wd_AE_features = wd_AE_features.detach()


    fig = plt.figure()
    fig.suptitle(f"PCA and Autoencoders")
    plt.tight_layout()

    plt.subplot(231)
    scatter1 = plt.scatter(AE_features[:, 0].cpu(), AE_features[:, 1].cpu(), **sc_kwargs)
    # legend1 = ax1.legend(*scatter1.legend_elements(), title="labels")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"AE with MSE loss")

    plt.subplot(232)
    scatter2 = plt.scatter(PCA_features[:, 0].cpu(), PCA_features[:, 1].cpu(), **sc_kwargs)
    # legend2 = ax2.legend(*scatter2.legend_elements(), title="labels")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PCA")

    plt.subplot(233)
    scatter3 = plt.scatter(wd_AE_features[:, 0].cpu(), wd_AE_features[:, 1].cpu(), **sc_kwargs)
    # legend2 = ax2.legend(*scatter2.legend_elements(), title="labels")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("AE with MSE and WD loss")

    plt.subplot(212)
    plt.errorbar(mean_AE_distances, [0], xerr=std_AE_distances, label="vanilla AE", fmt=".", capsize=4.)
    plt.errorbar(mean_wd_AE_distances, [0], xerr=std_wd_AE_distances, label="AE with WD", fmt=".", capsize=4.)
    plt.xlabel(r"$\langle distance \rangle$")
    plt.title(r"$\langle distance \rangle$ in latent space")
    plt.legend(loc="upper right")

    plt.savefig(f"/export/home/pnazari/workspace/AutoEncoderVisualization/AEPCA/output/comparison/comparison_wd={Decimal(weight_decay):.4e}.png")
    plt.show()
