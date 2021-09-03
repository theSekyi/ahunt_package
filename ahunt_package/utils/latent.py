import numpy as np
import torch
from pdb import set_trace
from torchvision.datasets import ImageFolder
from .base import (
    get_transforms,
    color,
    n_most_anomalous_images,
    rws_score,
    get_true_and_pred_labels,
    fraction_of_anomalies,
    ImageFolderWithPaths,
    balance_classes,
    tensorify_img,
    device_cpu,
)
from .file_manipulation import add_anomalies_to_training, get_file_len, dict_to_df
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
from train.training import training_loop
from pathlib import Path
from sklearn.ensemble import IsolationForest


from sklearnex import patch_sklearn


def get_layer_activation(model, name, layer, transform, path, device):
    """Returns Layer Activation"""
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    layer.register_forward_hook(get_activation(name))
    img = tensorify_img(path, transform)

    output = model(img[None, ...].float()).to(device)

    return activation[name].reshape(-1)


def get_latent_activations(model, name, layer, transform, ds, device):
    """Returns Latent Activations"""

    model = model.to(device_cpu)
    activations = []
    img_paths = [x[0] for x in ds.samples]
    img_labels = [x[1] for x in ds.samples]

    for path in img_paths:
        activation = get_layer_activation(model, name, layer, transform, path, device)
        activations.append(activation)
    return activations, img_labels, img_paths


def results_from_latent(model, train_ds, test_ds, transform, anomaly_label, device, layer):

    name = "latent"

    test_activations, test_labels, test_paths = get_latent_activations(model, name, layer, transform, test_ds, device)
    train_activations, train_labels, _ = get_latent_activations(model, name, layer, transform, train_ds, device)
    train_activations = [x.tolist() for x in train_activations]
    test_activations = [x.tolist() for x in test_activations]

    iforest = IsolationForest()
    iforest.fit(np.array(train_activations))

    scores_decision = iforest.decision_function(test_activations)  # the lower, the more anomalous

    scores_decision = scores_decision.max() - scores_decision

    labels_np = np.array([True if Path(x).parent.name == anomaly_label else False for x in test_paths])

    return test_paths, labels_np, scores_decision


def _get_latent(
    i,
    testing_path,
    train,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    saved_model_path,
    train_latent_path,
    batch_size_train,
    batch_size_test,
    anomaly_label,
    custom_model,
    device,
):
    print(f"{color.PURPLE}Latent Space for Night {i +1}{color.END}")
    _transform, _test_transform = get_transforms()

    train_ds = ImageFolder(train_latent_path, transform=_transform)
    test_ds = ImageFolderWithPaths(testing_path, transform=_test_transform)
    sampler = balance_classes(train_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=batch_size_train, drop_last=True)
    _ = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

    _, network_saved = training_loop(
        n_epoch, optim, Net, train_loader, learning_rate, momentum, saved_model_path, train, custom_model, device
    )

    test_paths, labels_np, scores_decision = results_from_latent(
        network_saved, train_ds, test_ds, _test_transform, anomaly_label, device, layer=network_saved.fc1
    )

    num_to_display = np.sum(labels_np)
    selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)

    add_anomalies_to_training(selected_anomalies, train_latent_path)

    y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    _latent_rws = rws_score(labels_np, scores_decision)
    _latent_mcc = matthews_corrcoef(y_true, y_pred)
    _latent_recall = recall_score(y_true, y_pred)
    _latent_precision = precision_score(y_true, y_pred)
    _latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    dict_to_df(get_file_len(train_latent_path))

    return _latent_rws, _latent_mcc, _latent_recall, _latent_precision, _latent_frac
