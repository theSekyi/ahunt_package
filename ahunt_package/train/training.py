import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
from pandas.core.common import flatten
from utils.base import color

from natsort import os_sorted

from torch import optim


def load_model(arch, path):
    """Load saved model"""
    network = arch()
    network.load_state_dict(torch.load(path))
    return network


def load_optim(optim, model, path, learning_rate, momentum):
    """Load Optimizer"""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.load_state_dict(path)
    return optimizer


def get_transforms():
    import torchvision.transforms as transforms

    _transform = transforms.Compose(
        [
            # transforms.RandomApply(
            #     [
            #         transforms.ColorJitter(),
            #         transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
            #         # transforms.RandomErasing(),
            #     ],
            #     p=0.3,
            # ),
            transforms.RandomAffine(10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ]
    )
    _test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return _transform, _test_transform


def test(network, test_loader, anomaly_label, anomaly_idx, device):

    label = []
    fl_paths = []
    scores_decision = []

    network.eval()
    network.to(device)

    with torch.no_grad():
        for i, (data, target, paths) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = network(data)
            _output = torch.exp(output)  # get probabilities

            _score_decision = _output[:, anomaly_idx]  # get probability of anomalous class

            _labels = [Path(lbl).parent.name for lbl in paths]

            _paths = list(paths)
            scores_decision.append(_score_decision.tolist())

            label.append(_labels)
            fl_paths.append(_paths)

    label_flat = list(flatten(label))

    labels_np = np.array([True if x == anomaly_label else False for x in label_flat])
    fl_paths = list(flatten(fl_paths))
    scores_decision = list(flatten(scores_decision))

    return fl_paths, labels_np, scores_decision


def train(epoch, optimizer, network, train_loader, saved_model_path, device):
    network.to(device)
    network.train()

    n_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_losses = loss.item()
        print(f"\t Batch {batch_idx + 1}/{n_batches} | Training Loss {train_losses}")

    torch.save(network.state_dict(), saved_model_path)
    torch.save(optimizer.state_dict(), "data/mnist/results/optimizer.pth")
    return train_losses, network


def get_network_and_optimizer(Net, optimizer, learning_rate, momentum, custom_model=True):
    if custom_model:
        network = Net()
    else:
        network = Net
    optimizer = optimizer.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer


def training_loop(
    n_epoch, optimizer, Net, train_loader, learning_rate, momentum, saved_model_path, train_func, custom_model, device
):
    network, optimizer = get_network_and_optimizer(Net, optimizer, learning_rate, momentum, custom_model)
    train_losses = []
    for epoch in range(1, n_epoch):
        print(f"{color.BOLD}Epoch {epoch}/{n_epoch - 1} {color.END}")
        train_loss, model = train_func(epoch, optimizer, network, train_loader, saved_model_path, device)
        train_losses.append(train_loss)
    return train_losses, model
