import matplotlib.pyplot as plt
import numpy as np


def plot_loss(epoch_nums, training_loss, validation_loss):
    plt.figure(figsize=(8, 8))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["training", "validation"], loc="upper right")
    plt.show()


def plot_single_loss(epoch_nums, loss, fig_size=(8, 8)):
    x = np.arange(epoch_nums - 1)

    plt.figure(figsize=fig_size)
    plt.plot(x, loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def plot_losses(
    epoch_nums,
    training_loss,
    validation_loss,
    training_loss_focal,
    validation_loss_focal,
):
    plt.figure(figsize=(8, 8))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.plot(epoch_nums, training_loss_focal)
    plt.plot(epoch_nums, validation_loss_focal)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(
        [
            "training cross-entropy loss",
            "validation cross-entropy loss",
            "training focal loss",
            "validation focal loss",
        ],
        loc="upper left",
    )
    plt.show()