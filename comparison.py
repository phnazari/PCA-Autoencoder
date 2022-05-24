import pickle
import shutil

import torch
import torchvision
from matplotlib import pyplot as plt

from evaluate import evaluate_model
from train import train_model
from models import LinearAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

test_dataset = torchvision.datasets.MNIST(root="/export/ial-nfs/user/user/data",
                                          train=False,
                                          transform=transform,
                                          download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

pca_model_path = f"/export/home/user/models/AutoEncoderVisualization/LinearAutoEncoder_pca_weights.pth"
model_path = f"/export/home/user/models/AutoEncoderVisualization/LinearAutoEncoder_weights.pth"

# touch those two
train = True
eval_model = True

# don't touch this one
load_model = not train

print(f"[Initialize] models")
# pca_model = LinearAutoEncoder(input_shape=784, load=load_model, path=pca_model_path).to(device)
model = LinearAutoEncoder(input_shape=784, load=True, path=model_path).to(device)

"""
Do PCA
"""


# good start: 1e-5

def comparison_plot():
    with open("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/data/distances.pkl", "rb") as f:
        distances_dict = pickle.load(f)

    smallest_n = torch.topk(torch.tensor(distances_dict["mean"]), 1, largest=False)

    weight_decay = distances_dict["wd"][smallest_n.indices[0]]

    filename = f"/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/comparison/comparison_wd={weight_decay}.png"
    shutil.copyfile(filename, "/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/comparison.png")


def hyperparam_opt():
    max_weight_decay = 1e-3
    min_weight_decay = 1e-5
    num_steps = 300

    weight_decay_rates = torch.linspace(min_weight_decay, max_weight_decay, num_steps)

    # reset file
    open("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/data/best.pkl", "wb").close()
    open("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/data/distances.pkl", "wb").close()
    for i, weight_decay in enumerate(weight_decay_rates):
        print(f"[Opt]: {i + 1} of {num_steps}")

        # initialize new model
        pca_model = LinearAutoEncoder(input_shape=784, load=False).to(device)

        # train and evaluation.py model
        train_model(pca_model, pca_model_path, epochs=50, weight_decay=weight_decay.item())
        evaluate_model(model, test_loader, pca_model, weight_decay.item())

    # do final hyperparameter-opt analysis
    with open("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/data/distances.pkl", "rb") as f:
        distances_dict = pickle.load(f)

    """
    Plotting
    """

    n = 20
    smallest_n = torch.topk(torch.tensor(distances_dict["mean"]), n, largest=False)
    with open("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/data/best.pkl", "wb") as f:
        pickle.dump(smallest_n, f)

    print(smallest_n)

    fig = plt.figure()

    fig.suptitle("l2 distance between AE and PCA")

    plt.errorbar(distances_dict["wd"], distances_dict["mean"], yerr=distances_dict["std"],
                 marker=".",
                 color="navy",
                 capsize=4.)

    plt.xlabel("weight decay rate")
    # plt.xscale("log")
    plt.ylabel("mean distance")
    plt.savefig("/export/home/user/workspace/AutoEncoderVisualization/AEPCA/output/distances.png")

    plt.show()


hyperparam_opt()
comparison_plot()
