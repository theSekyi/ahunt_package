from .base import (
    get_transforms,
    color,
    n_most_anomalous_images,
    rws_score,
    get_true_and_pred_labels,
    fraction_of_anomalies,
)
from .file_manipulation import add_anomalies_to_training, get_file_len, dict_to_df

from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.ensemble import IsolationForest
from PIL import Image
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
from pathlib import Path


def get_np_imgs(ds):
    lst_imgs = []
    lst_labels = []
    lst_path = []

    for pth, lbl in ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    np_imgs = np.array(lst_imgs)
    labels = np.array(lst_labels)

    return np_imgs, labels, lst_path


def get_np_imgs_all(train_ds, test_ds):
    lst_imgs = []
    lst_labels = []
    lst_path = []

    for pth, lbl in train_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    for pth, lbl in test_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    np_imgs = np.array(lst_imgs)
    labels = np.array(lst_labels)

    return np_imgs, labels, lst_path


def results_from_iforest(train_ds, test_ds, iforest_use_all, anomaly_label):
    from pathlib import Path

    train_np_imgs, _, _ = get_np_imgs(train_ds)
    test_np_imgs, _, test_paths = get_np_imgs(test_ds)
    np_imgs_all, _, _ = get_np_imgs_all(train_ds, test_ds)

    iforest = IsolationForest()

    if iforest_use_all:
        iforest.fit(np_imgs_all)
    else:
        iforest.fit(train_np_imgs)

    scores_decision = iforest.decision_function(test_np_imgs)

    scores_decision = scores_decision.max() - scores_decision

    labels_np = np.array([True if Path(x).parent.name == anomaly_label else False for x in test_paths])
    return labels_np, scores_decision, test_paths


def _get_iforest(i, testing_path, train_iforest_path, anomaly_label, iforest_use_all):
    _transform, _test_transform = get_transforms()

    print(f"{color.PURPLE}Iforest for Night {i+1}{color.END}")

    train_ds, test_ds = (
        ImageFolder(train_iforest_path, transform=_transform),
        ImageFolder(testing_path, transform=_transform),
    )

    (
        labels_np,
        scores_decision,
        test_paths,
    ) = results_from_iforest(train_ds, test_ds, iforest_use_all, anomaly_label)

    num_to_display = np.sum(labels_np)

    selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)

    add_anomalies_to_training(selected_anomalies, train_iforest_path)

    y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    _iso_rws = rws_score(labels_np, scores_decision)
    _iso_mcc = matthews_corrcoef(y_true, y_pred)
    _iso_recall = recall_score(y_true, y_pred)
    _iso_precision = precision_score(y_true, y_pred)
    _iso_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    dict_to_df(get_file_len(train_iforest_path))

    return _iso_rws, _iso_mcc, _iso_recall, _iso_precision, _iso_frac
