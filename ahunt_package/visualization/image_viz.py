import matplotlib.pyplot as plt
from utils import balance_classes
from training import get_transforms
from torchvision.datasets import ImageFolder
import torch
from glob import glob
import matplotlib as mpl
import random


def find_idx_multiples(N):
    multiples = [x for x in range(1, N + 1) if N % x == 0]
    len_multiples = len(multiples)
    midpoint = len_multiples / 2
    idx_1, idx_2 = int(midpoint - 0.5), int(midpoint - 0.5)
    if len_multiples % 2 == 0:
        idx_1, idx_2 = int(midpoint), int(midpoint - 1)
    num_1, num_2 = multiples[idx_1], multiples[idx_2]
    return num_1, num_2


def plot_batch(train_path, batch_size):
    _transform, _test_transform = get_transforms()

    train_ds = ImageFolder(train_path, transform=_transform)
    sampler = balance_classes(train_ds)
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=batch_size, drop_last=True)

    image_tensors, labels = next(iter(train_loader))
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {class_to_idx[i]: i for i in class_to_idx}
    nrows, ncols = find_idx_multiples(batch_size)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
    for (image_tensor, label, ax) in zip(image_tensors, labels, axes.flatten()):
        image_tensor = image_tensor.permute(1, 2, 0)
        image_class = str(idx_to_class[label.item()])
        ax.imshow(image_tensor, cmap="bone", vmin=-1.5, vmax=1.5)
        ax.set_title(image_class)
        ax.axis("off")
    plt.show()


def get_image_paths(pattern):
    return glob(pattern)


def get_rand_img(num_to_select, img_list):
    return random.sample(img_list, num_to_select)


def plot_grid(row, col, img_list):
    fig = plt.figure(figsize=(12, 15))

    for i, img in enumerate(img_list):
        ax = fig.add_subplot(row, col, i + 1)
        ax.axis("off")
        img = plt.imread(img)
        plt.imshow(img)
    plt.show()


def plot_random_sample(pattern, num_to_select, row_no, col_no, c_map="viridis"):
    """Plot sample images from a folder"""
    mpl.rc("image", cmap=c_map)
    all_images = get_image_paths(pattern)
    sampled_img = get_rand_img(num_to_select, all_images)
    plot_grid(row_no, col_no, sampled_img)
