from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import pathlib
import skimage
import skimage.io
import matplotlib.pyplot as plt
import collections


def get_file_len(path):

    if path.endswith("/"):  # check if path doesn't end in /
        path = path[:-1]

    iter_fl = iter(os.walk(path))
    dictn = {}
    next(iter_fl)
    for r, d, files in iter_fl:
        new_r = os.path.basename(r)

        if not new_r.startswith("."):  # ignore hidden directories ie.eg .ipython_checkpoints
            dictn[new_r] = len(files)
    return dictn


def dict_to_df(dictn):
    """Convert a dictionary to a nicely formatted table"""
    df = pd.DataFrame(dictn.items(), columns=["Directory", "Number of Files"])
    print(tabulate(df, headers=df.columns, tablefmt="fancy_grid", showindex="never"))


def add_anomalies_to_training(selected_anomalies, dest_path):

    for path in selected_anomalies:
        label = Path(path).parent.name
        dest_folder = f"{dest_path}/{label}"
        shutil.copy2(path, dest_folder)


def save_list(my_lst, fl_name):
    np.save(fl_name, my_lst)


def load_lst(fl_name):
    return np.load(fl_name, allow_pickle=True).tolist()


def move_files(pth, destination_path):
    path_to_labels = [f for f in pth.iterdir() if f.is_dir()]

    for label_path in path_to_labels:
        fl_names = os.listdir(label_path)
        label_name = label_path.name
        dest_path = f"{destination_path}/{label_name}"

        for fl_name in fl_names:
            shutil.move(os.path.join(label_path, fl_name), dest_path)


def initialize_pool(src, dest):
    """Initialize the data Pool. Take data from source to the pool"""
    shutil.rmtree(dest)
    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def rm_folders(*args):
    """Empty Directories"""
    for arg in args:
        dir_path = pathlib.Path(arg)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
        pathlib.Path(arg).mkdir(parents=True, exist_ok=True)


def zero_folders(*args, src, dest):
    """Remove folders specifies in args and copy data from source to pool(dest)"""
    rm_folders(*args)

    initialize_pool(src, dest)


def get_basic_config(rounds, data_config, anomaly_label, constant_anomaly_size=True):
    all_data_config = []
    for i in range(1, rounds + 1):
        if not constant_anomaly_size:
            data = {a: (int((v * i) / 2) if a == anomaly_label else v) for a, v in data_config.items()}
        else:
            data = data_config
        # if i == 1:
        #     data = {a: (0 if a == anomaly_label else v) for a, v in data_config.items()}
        all_data_config.append(data)
    return all_data_config


def np_to_img(pth_to_save, np_array):
    return plt.imsave(pth_to_save, np_array, cmap="Greys")


def add_noise(pth, var):

    origin = skimage.io.imread(pth)
    noisy = skimage.util.random_noise(origin, mode="gaussian", var=var)

    return noisy


def cp_and_add_noise(src, dest, file_names, var):
    for fl_name in file_names:
        img_src_path = os.path.join(src, fl_name)
        noisy_img = add_noise(img_src_path, var)
        dest_img = f"{dest}/{fl_name}"
        np_to_img(dest_img, noisy_img)


def _get_individual_training_paths(training_path):
    iforest_path = f"{training_path}iforest/"
    latent_path = f"{training_path}latent/"
    ahunt_path = f"{training_path}ahunt/"

    return iforest_path, latent_path, ahunt_path


def create_dir_training(training_path):
    iforest_path, latent_path, ahunt_path = _get_individual_training_paths(training_path)
    dir_to_create = [iforest_path, latent_path, ahunt_path]
    for path in dir_to_create:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return iforest_path, latent_path, ahunt_path


def get_random_fls(path, percentage=None, num_files=None):
    files = [
        os.path.join(str(path), f)
        for f in os.listdir(path)
        if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".JPEG", "jpeg"))
    ]
    if num_files:
        random_files = np.random.choice(files, int(num_files), replace=False)
    else:
        random_files = np.random.choice(files, int(len(files) * percentage), replace=False)
    fl_names = [x.split("/")[-1] for x in random_files]

    return random_files, fl_names


def cp_files(src, dest, fl_names):
    import shutil

    for fl in fl_names:
        shutil.copy(os.path.join(src, fl, dest))


def rm_files(src, fl_names):
    for fl in fl_names:
        pathlib.Path(os.path.join(src, fl)).unlink()


def initial_training(src_pool, training_path, init_config, var, add_noise=False):
    # Creates folders for Iforest,Latent and Ahunt
    iforest_path, latent_path, ahunt_path = create_dir_training(training_path)

    for folder, size in init_config.items():

        label_src = f"{src_pool}/{folder}"

        iso_dest = f"{iforest_path}/{folder}"
        latent_dest = f"{latent_path}/{folder}"
        ahunt_dest = f"{ahunt_path}/{folder}"

        pathlib.Path(iso_dest).mkdir(parents=True, exist_ok=True)
        pathlib.Path(latent_dest).mkdir(parents=True, exist_ok=True)
        pathlib.Path(ahunt_dest).mkdir(parents=True, exist_ok=True)

        if size != 0:
            _, fl_names = get_random_fls(label_src, num_files=size)

            if add_noise:
                cp_and_add_noise(label_src, iso_dest, fl_names, var=var)
                cp_and_add_noise(label_src, latent_dest, fl_names, var=var)
                cp_and_add_noise(label_src, ahunt_dest, fl_names, var=var)
            else:
                cp_files(label_src, iso_dest, fl_names)
                cp_files(label_src, latent_dest, fl_names)
                cp_files(label_src, ahunt_dest, fl_names)

            rm_files(label_src, fl_names)
    print("Iforest")
    dict_to_df(get_file_len(iforest_path))
    print("Latent")
    dict_to_df(get_file_len(latent_path))
    print("Ahunt")
    dict_to_df(get_file_len(ahunt_path))


def mv_files(src, dest, fl_names):
    for fl in fl_names:
        shutil.move(os.path.join(src, fl), dest)


def create_ds_all_config(path, src, all_data_config, var=None, add_noise=False, round_name="round"):

    for i, data_config in enumerate(all_data_config):
        round_folder = f"{path}/{round_name}_{i}"
        pathlib.Path(round_folder).mkdir(parents=True, exist_ok=True)
        for folder, data_size in data_config.items():  # create folder for labels
            dest = f"{round_folder}/{folder}"
            label_src = f"{src}/{folder}"

            pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
            if data_size != 0:
                _, file_names = get_random_fls(label_src, num_files=data_size)
                if add_noise:
                    cp_and_add_noise(label_src, dest, file_names, var=var)
                else:
                    mv_files(label_src, dest, file_names)
                rm_files(label_src, file_names)  # pick without replacement
        dict_to_df(get_file_len(round_folder))


def create_test_folders(initial_test_config, test_main):
    rm_folders(f"{test_main}/testing_main")
    lst_folders = ["iforest", "latent", "ahunt"]
    labels = initial_test_config.keys()
    for i in lst_folders:
        fld = f"{test_main}/testing_main/{i}"
        pathlib.Path(fld).mkdir(parents=True, exist_ok=True)

        for label in labels:
            fld_label = f"{fld}/{label}"
            pathlib.Path(fld_label).mkdir(parents=True, exist_ok=True)


def basic_reset_medmnist(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/medmnist/medical-mnist"
    dest_pool = "data/medmnist/pool"
    medmnist_test = "data/medmnist/testing"
    src = "data/medmnist/pool"
    medmnist_main = "data/medmnist"

    zero_folders(
        "data/medmnist/pool",
        "data/medmnist/testing",
        "data/medmnist/training",
        "data/medmnist/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/medmnist/training/"
    initial_training(
        "data/medmnist/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(medmnist_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, medmnist_main)


def basic_reset_mnist(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/mnist/mnist/training"
    dest_pool = "data/mnist/pool"
    mnist_test = "data/mnist/testing"
    src = "data/mnist/pool"
    mnist_main = "data/mnist"

    zero_folders(
        "data/mnist/pool",
        "data/mnist/testing",
        "data/mnist/training",
        "data/mnist/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/mnist/training/"
    initial_training(
        "data/mnist/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(mnist_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, mnist_main)


def basic_reset_galaxy(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/galaxy_zoo/g_zoo"
    dest_pool = "data/galaxy_zoo/pool"
    galaxy_test = "data/galaxy_zoo/testing"
    src = "data/galaxy_zoo/pool"
    galaxy_main = "data/galaxy_zoo"

    zero_folders(
        "data/galaxy_zoo/pool",
        "data/galaxy_zoo/testing",
        "data/galaxy_zoo/training",
        "data/galaxy_zoo/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/galaxy_zoo/training/"
    initial_training(
        "data/galaxy_zoo/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(galaxy_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, galaxy_main)


def create_label_folders(img_folder_path, labels):
    "Creates Folders for Labels"
    from pathlib import Path

    folders = ["train", "test"]

    for folder in folders:
        for label in labels:
            Path(f"{img_folder_path}/{folder}/{label}").mkdir(parents=True, exist_ok=True)


def get_all_folder_files(path):
    import os

    files = [
        os.path.join(str(path), f)
        for f in os.listdir(path)
        if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".JPEG", "jpeg"))
    ]
    return files


def get_file_split(files, split=0.8):

    import numpy as np

    train_split = np.random.choice(files, int(len(files) * split), replace=False).tolist()
    test_split = list(set(files) - set(train_split))

    return train_split, test_split


def create_train_and_test(src, labels, split=0.8):
    import collections

    split_dict = collections.defaultdict(list)

    for label in labels:
        label_files = get_all_folder_files(f"{src}/{label}")
        train_split, test_split = get_file_split(label_files)
        split_dict[label] = [train_split, test_split]

    return split_dict


def get_folders_in_dir(path):
    from glob import glob

    all_folders = glob(f"{path}/**/**/")
    return all_folders


def cp_files_from_path(file_paths, dest):
    import shutil

    for fl in file_paths:
        shutil.copy(fl, dest)


def move_train_test_split(split_dict, dest):
    flds = get_folders_in_dir(dest)
    for fld in flds:
        fld_path = Path(fld)
        if fld_path.parent.name == "train":
            label_name = fld_path.name
            training_set = split_dict[label_name][0]
            cp_files_from_path(training_set, fld_path)
        if fld_path.parent.name == "test":
            label_name = fld_path.name
            test_set = split_dict[label_name][1]
            cp_files_from_path(test_set, fld_path)


def is_test_fl_unique(test_pth):
    """Checks if all images in test are unique"""

    all_data = collections.defaultdict(list)
    label_names = [f.name for f in test_pth[0].iterdir() if f.is_dir()]

    for pth in test_pth:
        for label in label_names:
            pth_label = f"{pth}/{label}"
            fl_names = os.listdir(pth_label)
            all_data[label].extend(fl_names)

    for k, v in all_data.items():
        if len(v) == len(set(v)):
            print(f"Label {k} is unique")
        else:
            print(f"Label {k} isn't unique")