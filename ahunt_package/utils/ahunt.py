import torch
from torchvision.datasets import ImageFolder
import numpy as np

from .base import (
    get_transforms,
    color,
    n_most_anomalous_images,
    rws_score,
    get_true_and_pred_labels,
    fraction_of_anomalies,
    ImageFolderWithPaths,
    balance_classes,
    get_anomaly_index,
)
from .file_manipulation import add_anomalies_to_training, get_file_len, dict_to_df
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
from train.training import training_loop, test


def _get_ahunt(
    i,
    testing_path,
    train,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    saved_model_path,
    train_ahunt_path,
    batch_size_train,
    batch_size_test,
    anomaly_label,
    custom_model,
    device,
):
    print(f"{color.PURPLE}Ahunt for Night {i +1}{color.END}")
    _transform, _test_transform = get_transforms()

    train_ds = ImageFolder(train_ahunt_path, transform=_transform)
    sampler = balance_classes(train_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=batch_size_train, drop_last=True)

    _, network_saved = training_loop(
        n_epoch, optim, Net, train_loader, learning_rate, momentum, saved_model_path, train, custom_model, device
    )

    test_ds = ImageFolderWithPaths(testing_path, transform=_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

    anomaly_idx = get_anomaly_index(test_loader, anomaly_label)

    test_paths, labels_np, scores_decision = test(network_saved, test_loader, anomaly_label, anomaly_idx, device)

    num_to_display = np.sum(labels_np)

    selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)

    add_anomalies_to_training(selected_anomalies, train_ahunt_path)

    y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    _ahunt_rws = rws_score(labels_np, scores_decision)
    _ahunt_mcc = matthews_corrcoef(y_true, y_pred)
    _ahunt_recall = recall_score(y_true, y_pred)
    _ahunt_precision = precision_score(y_true, y_pred)
    _ahunt_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    dict_to_df(get_file_len(train_ahunt_path))

    return _ahunt_rws, _ahunt_mcc, _ahunt_recall, _ahunt_precision, _ahunt_frac
