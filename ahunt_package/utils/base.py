import torch

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from torchvision.datasets import ImageFolder


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")
from .file_manipulation import dict_to_df, get_file_len, move_files
from .isolation_forests import _get_iforest
from .latent import _get_latent
from .ahunt import _get_ahunt


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def free_gpu():
    return torch.cuda.empty_cache()


def rws_score(is_outlier, outlier_score, n_o=None):
    outliers = np.array(is_outlier)
    if n_o is None:
        n_o = int(np.sum(outliers))
    b_s = np.arange(n_o) + 1
    o_ind = np.argsort(outlier_score)[-n_o:]
    if b_s.size == 0:
        return 0

    return 1.0 * np.sum(b_s * outliers[o_ind].reshape(-1)) / np.sum(b_s)


def get_true_and_pred_labels(labels_np, scores_decision):
    n_anomalies = np.sum(labels_np)

    threshold = np.sort(scores_decision)[-n_anomalies - 1]

    y_pred = scores_decision > threshold
    y_true = labels_np.astype(int)

    return y_true, y_pred


def get_anomaly_index(test_loader, anomaly_label):
    class_to_idx = test_loader.dataset.class_to_idx
    for x, y in class_to_idx.items():
        if x == anomaly_label:
            return y
    return "Anomaly class not found"


def n_most_anomalous_images(scores_decision, test_paths, num_to_display):

    _, _anomalous_paths = zip(*sorted(zip(scores_decision, test_paths), reverse=True))

    selected_anomalies = _anomalous_paths[:num_to_display]

    return selected_anomalies


def fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label):

    total_anomalies = sum(labels_np)
    if total_anomalies == 0:
        return 0

    num_anomalies_detected = 0
    for path in selected_anomalies:
        _label = Path(path).parent.name
        if _label == anomaly_label:
            num_anomalies_detected += 1

    return num_anomalies_detected / total_anomalies


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def balance_classes(ds):
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import WeightedRandomSampler

    target = np.array(ds.targets)
    cls_weights = torch.from_numpy(compute_class_weight("balanced", np.unique(target), target))
    weights = cls_weights[torch.from_numpy(target)]
    sampler = WeightedRandomSampler(weights, len(target), replacement=True)
    return sampler


def is_grayscale(path, transform):
    return transform(Image.open(path)).shape[0] == 1


def tensorify_img(path, transform):
    img = Image.open(path)

    if is_grayscale(path, transform):
        img = Image.open(path).convert("RGB")

    img_tensor = transform(img)
    return img_tensor


def get_combined_output(
    test_pth,
    iforest_use_all,
    train_iforest_path,
    train_latent_path,
    train_ahunt_path,
    batch_size_train,
    batch_size_test,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    train,
    saved_model_path,
    anomaly_label,
    collective_test_path,
    use_cummulative_test,
    custom_model,
    device,
):
    iso_rws = []
    iso_mcc = []
    iso_recall = []
    iso_precision = []
    iso_frac = []

    latent_rws = []
    latent_mcc = []
    latent_recall = []
    latent_precision = []
    latent_frac = []

    ahunt_rws = []
    ahunt_mcc = []
    ahunt_recall = []
    ahunt_precision = []
    ahunt_frac = []

    for i, pth in enumerate(test_pth):

        if use_cummulative_test:
            testing_path = collective_test_path
            move_files(pth, testing_path)

        else:
            testing_path = pth

        dict_to_df(get_file_len(testing_path))

        _iso_rws, _iso_mcc, _iso_recall, _iso_precision, _iso_frac = _get_iforest(
            i, testing_path, train_iforest_path, anomaly_label, iforest_use_all
        )
        iso_rws.append(_iso_rws)
        iso_mcc.append(_iso_mcc)
        iso_recall.append(_iso_recall)
        iso_precision.append(_iso_precision)
        iso_frac.append(_iso_frac)

        _latent_rws, _latent_mcc, _latent_recall, _latent_precision, _latent_frac = _get_latent(
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
        )
        latent_mcc.append(_latent_mcc)
        latent_rws.append(_latent_rws)
        latent_recall.append(_latent_recall)
        latent_precision.append(_latent_precision)
        latent_frac.append(_latent_frac)

        _ahunt_rws, _ahunt_mcc, _ahunt_recall, _ahunt_precision, _ahunt_frac = _get_ahunt(
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
        )

        ahunt_rws.append(_ahunt_rws)
        ahunt_mcc.append(_ahunt_mcc)
        ahunt_recall.append(_ahunt_recall)
        ahunt_precision.append(_ahunt_precision)
        ahunt_frac.append(_ahunt_frac)

    return (
        iso_rws,
        iso_mcc,
        iso_recall,
        iso_precision,
        iso_frac,
        latent_rws,
        latent_mcc,
        latent_recall,
        latent_precision,
        latent_frac,
        ahunt_rws,
        ahunt_mcc,
        ahunt_recall,
        ahunt_precision,
        ahunt_frac,
    )
