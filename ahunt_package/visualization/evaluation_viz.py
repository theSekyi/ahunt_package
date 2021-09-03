import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
import itertools


def _plt_ax(ax, x, y, color, label):
    ax.plot(x, y, f"{color}-", label=label)
    ax.legend()


def _plt_latent_individual(
    ax,
    x,
    y1,
    y2,
    y3,
    y4,
    color_1,
    color_2,
    color_3,
    color_4,
    label_1,
    label_2,
    label_3,
    label_4,
    y_label,
):
    _plt_ax(ax, x, y1, color_1, label_1)
    _plt_ax(ax, x, y2, color_2, label_2)
    _plt_ax(ax, x, y3, color_3, label_3)
    _plt_ax(ax, x, y4, color_4, label_4)
    ax.set_xlabel("Rounds", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)


def plt_latent(
    fc1_latent_rws,
    fc1_latent_mcc,
    fc1_latent_recall,
    fc1_latent_precision,
    fc1_latent_frac,
    fc2_latent_rws,
    fc2_latent_mcc,
    fc2_latent_recall,
    fc2_latent_precision,
    fc2_latent_frac,
    fc3_latent_rws,
    fc3_latent_mcc,
    fc3_latent_recall,
    fc3_latent_precision,
    fc3_latent_frac,
    fc4_latent_rws,
    fc4_latent_mcc,
    fc4_latent_recall,
    fc4_latent_precision,
    fc4_latent_frac,
):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.tight_layout(pad=4)

    x = np.arange(len(fc1_latent_rws))

    fig.suptitle(
        f"Performance of different Layers in Latent Space over {len(x)} rounds of training",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.94)

    _plt_latent_individual(
        ax1,
        x,
        fc1_latent_rws,
        fc2_latent_rws,
        fc3_latent_rws,
        fc4_latent_rws,
        "r",
        "g",
        "b",
        "c",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        y_label="RWS Score",
    )

    _plt_latent_individual(
        ax2,
        x,
        fc1_latent_frac,
        fc2_latent_frac,
        fc3_latent_frac,
        fc4_latent_frac,
        "r",
        "g",
        "b",
        "c",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        y_label="Fraction of Anomalies Detected",
    )

    _plt_latent_individual(
        ax3,
        x,
        fc1_latent_mcc,
        fc2_latent_mcc,
        fc3_latent_mcc,
        fc4_latent_mcc,
        "r",
        "g",
        "b",
        "c",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        y_label="MCC",
    )

    _plt_latent_individual(
        ax4,
        x,
        fc1_latent_recall,
        fc2_latent_recall,
        fc3_latent_recall,
        fc4_latent_recall,
        "r",
        "g",
        "b",
        "c",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        y_label="Recall",
    )

    _plt_latent_individual(
        ax5,
        x,
        fc1_latent_precision,
        fc2_latent_precision,
        fc3_latent_precision,
        fc4_latent_precision,
        "r",
        "g",
        "b",
        "c",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        y_label="Precision",
    )

    ax6.axis("off")

    sns.despine()
    plt.show()


def get_mean_n_std(sample):

    sample_len = len(sample)
    sample_std = np.std(sample, axis=0)
    sample_mean = np.sum(sample, axis=0) / sample_len

    return sample_std, sample_mean


def plt_fill_between(all_iso_rws, all_latent_rws, all_ahunt_rws, y_label):
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 8))

    x = np.arange(len(all_ahunt_rws[0]))

    iso_std, iso_mean = get_mean_n_std(all_iso_rws)
    plt.plot(x, iso_mean, "b-", label="Isolation Forest")
    plt.fill_between(
        x,
        iso_mean - iso_std,
        iso_mean + iso_std,
        color="b",
        alpha=0.25,
    )

    latent_std, latent_mean = get_mean_n_std(all_latent_rws)
    plt.plot(x, latent_mean, "r-", label="Latent Space")
    plt.fill_between(
        x,
        latent_mean - latent_std,
        latent_mean + latent_std,
        color="r",
        alpha=0.25,
    )

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_rws)
    plt.plot(x, ahunt_mean, "g-", label="Ahunt")
    plt.fill_between(
        x,
        ahunt_mean - ahunt_std,
        ahunt_mean + ahunt_std,
        color="g",
        alpha=0.25,
    )

    plt.xlabel("Night")
    plt.ylabel(y_label)

    plt.legend(loc="upper left")
    plt.show()


def _plt_fill(ax, x, mean, std, color, label):
    ax.plot(x, mean, f"{color}-", label=label)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.25,
    )
    ax.legend()


def plt_iso_fill_btn(
    all_iso_static_rws,
    all_iso_static_mcc,
    all_iso_static_recall,
    all_iso_static_precision,
    all_iso_static_frac,
    all_iso_learning_rws,
    all_iso_learning_mcc,
    all_iso_learning_recall,
    all_iso_learning_precision,
    all_iso_learning_frac,
    figsize=(12, 15),
):
    plt.style.use("seaborn-dark-palette")

    x = np.arange(len(all_iso_static_rws[0]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.tight_layout(pad=4)

    fig.suptitle(
        f"Iforest with dynamic training vs Iforest Static over {len(x)} rounds",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.94)  # Aligns title properly in the presence of fig.tight_layout

    ####### RWS!

    iso_std, iso_mean = get_mean_n_std(all_iso_static_rws)
    _plt_fill(ax1, x, iso_mean, iso_std, "b", "Static Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_iso_learning_rws)
    _plt_fill(ax1, x, latent_mean, latent_std, "r", "Dynamic Isolation Forest")

    ax1.set_xlabel("Rounds", fontsize=10)
    ax1.set_ylabel("RWS score", fontsize=10)

    ### MCC
    iso_std, iso_mean = get_mean_n_std(all_iso_static_mcc)
    _plt_fill(ax2, x, iso_mean, iso_std, "b", "Static Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_iso_learning_mcc)
    _plt_fill(ax2, x, latent_mean, latent_std, "r", "Dynamic Isolation Forest")

    ax2.set_xlabel("Rounds", fontsize=10)
    ax2.set_ylabel("MCC score", fontsize=10)

    ### Recall
    iso_std, iso_mean = get_mean_n_std(all_iso_static_recall)
    _plt_fill(ax3, x, iso_mean, iso_std, "b", "Static Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_iso_learning_recall)
    _plt_fill(ax3, x, latent_mean, latent_std, "r", "Dynamic Isolation Forest")

    ax3.set_xlabel("Rounds", fontsize=10)
    ax3.set_ylabel("Recall", fontsize=10)

    ### Precision
    iso_std, iso_mean = get_mean_n_std(all_iso_static_precision)
    _plt_fill(ax4, x, iso_mean, iso_std, "b", "Static Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_iso_learning_precision)
    _plt_fill(ax4, x, latent_mean, latent_std, "r", "Dynamic Isolation Forest")

    ax4.set_xlabel("Rounds", fontsize=10)
    ax4.set_ylabel("Precision", fontsize=10)

    ### Fraction
    iso_std, iso_mean = get_mean_n_std(all_iso_static_frac)
    _plt_fill(ax5, x, iso_mean, iso_std, "b", "Static Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_iso_learning_frac)
    _plt_fill(ax5, x, latent_mean, latent_std, "r", "Dynamic Isolation Forest")

    ax5.set_xlabel("Rounds", fontsize=10)
    ax5.set_ylabel("Fraction of anomalies detected", fontsize=10)

    ax6.axis("off")

    sns.despine()

    plt.show()


def multiple_plt_fill_between(
    all_iso_rws,
    all_iso_mcc,
    all_iso_frac,
    all_iso_recall,
    all_iso_precision,
    all_latent_rws,
    all_latent_mcc,
    all_latent_frac,
    all_latent_recall,
    all_latent_precision,
    all_ahunt_rws,
    all_ahunt_mcc,
    all_ahunt_frac,
    all_ahunt_recall,
    all_ahunt_precision,
    font_size=14,
    figsize=(8, 8),
):

    #     sns.set_style("darkgrid")
    #     plt.style.use('ggplot')
    plt.style.use("seaborn-dark-palette")

    x = np.arange(len(all_ahunt_rws[0]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.tight_layout(pad=4)

    fig.suptitle(f"Improvement of Ahunt after {len(all_ahunt_rws[0])} rounds of training", fontsize=14)
    fig.subplots_adjust(top=0.94)  # Aligns title properly in the presence of fig.tight_layout

    ## RWS

    iso_std, iso_mean = get_mean_n_std(all_iso_rws)
    _plt_fill(ax1, x, iso_mean, iso_std, "y", "Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_latent_rws)
    _plt_fill(ax1, x, latent_mean, latent_std, "g", "Latent Space")

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_rws)
    _plt_fill(ax1, x, ahunt_mean, ahunt_std, "b", "Ahunt")

    ax1.set_xlabel("Rounds", fontsize=font_size)
    ax1.set_ylabel("RWS score", fontsize=font_size)

    # MCC
    iso_std, iso_mean = get_mean_n_std(all_iso_mcc)
    _plt_fill(ax2, x, iso_mean, iso_std, "y", "Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_latent_mcc)
    _plt_fill(ax2, x, latent_mean, latent_std, "g", "Latent Space")

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_mcc)
    _plt_fill(ax2, x, ahunt_mean, ahunt_std, "b", "Ahunt")

    ax2.set_xlabel("Rounds", fontsize=font_size)
    ax2.set_ylabel("MCC Score", fontsize=font_size)

    # FRACTION
    iso_std, iso_mean = get_mean_n_std(all_iso_frac)
    _plt_fill(ax3, x, iso_mean, iso_std, "y", "Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_latent_frac)
    _plt_fill(ax3, x, latent_mean, latent_std, "g", "Latent Space")

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_frac)
    _plt_fill(ax3, x, ahunt_mean, ahunt_std, "b", "Ahunt")

    ax3.set_xlabel("Rounds", fontsize=font_size)
    ax3.set_ylabel("Fraction of anomalies", fontsize=font_size)

    ## Recall

    iso_std, iso_mean = get_mean_n_std(all_iso_recall)
    _plt_fill(ax4, x, iso_mean, iso_std, "y", "Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_latent_recall)
    _plt_fill(ax4, x, latent_mean, latent_std, "g", "Latent Space")

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_recall)
    _plt_fill(ax4, x, ahunt_mean, ahunt_std, "b", "Ahunt")

    ax4.set_xlabel("Rounds", fontsize=font_size)
    ax4.set_ylabel("Recall", fontsize=font_size)

    # ## Precision
    iso_std, iso_mean = get_mean_n_std(all_iso_precision)
    _plt_fill(ax5, x, iso_mean, iso_std, "y", "Isolation Forest")

    latent_std, latent_mean = get_mean_n_std(all_latent_precision)
    _plt_fill(ax5, x, latent_mean, latent_std, "g", "Latent Space")

    ahunt_std, ahunt_mean = get_mean_n_std(all_ahunt_precision)
    _plt_fill(ax5, x, ahunt_mean, ahunt_std, "b", "Ahunt")
    ax5.set_xlabel("Rounds", fontsize=font_size)
    ax5.set_ylabel("Precision", fontsize=font_size)

    ax6.axis("off")

    sns.despine()

    plt.show()


def plot_single_with_fill(lst_of_metric, label, color="g", fig_size=(8, 8)):
    x = np.arange(len(lst_of_metric[0]))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    fig.suptitle(label, fontsize=14)

    _std, _mean = get_mean_n_std(lst_of_metric)

    ax.plot(x, _mean, f"{color}-")
    ax.fill_between(
        x,
        _mean - _std,
        _mean + _std,
        color=color,
        alpha=0.25,
    )

    sns.despine()

    plt.show()


def save_list(my_lst, fl_name):
    np.save(fl_name, my_lst)


def load_lst(fl_name):
    import numpy as np

    return np.load(fl_name, allow_pickle=True).tolist()


def analyze(xx, cl=25):
    m = np.mean(xx, axis=0)
    l = np.percentile(xx, cl, axis=0)
    u = np.percentile(xx, 100 - cl, axis=0)
    return m, l, u


def a_plt_fill(ax, x, m, l, u, color, label):
    ax.plot(x, m, f"{color}-", label=label)
    ax.fill_between(x, l, u, color=color, alpha=0.2)
    ax.legend()


def a_plot_all_single_metric(rws, mcc, recall, precision, frac, lbl, fig_size=(8, 8)):
    x = np.arange(len(rws[0]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=fig_size)
    fig.suptitle(lbl, fontsize=14)

    m, l, u = analyze(rws, cl=2.5)
    a_plt_fill(ax1, x, m, l, u, "y", "RWS Score")
    # ax1.plot(x, rws_mean, "g-")
    # ax1.fill_between(x, rws_mean - rws_std, rws_mean + rws_std, color="g", alpha=0.25, label="RWS Score")
    # ax1.legend()

    m, l, u = analyze(mcc, cl=2.5)
    a_plt_fill(ax2, x, m, l, u, "b", "MCC")
    # mcc_std, mcc_mean = get_mean_n_std(mcc)
    # ax2.plot(x, rws_mean, "b-")
    # ax2.fill_between(x, mcc_mean - mcc_std, mcc_mean + mcc_std, color="b", alpha=0.25, label="MCC Score")
    # ax2.legend()

    m, l, u = analyze(recall, cl=2.5)
    a_plt_fill(ax3, x, m, l, u, "r", "Recall")

    # recall_std, recall_mean = get_mean_n_std(recall)
    # ax3.plot(x, recall_mean, "r-")
    # ax3.fill_between(x, recall_mean - recall_std, recall_mean + recall_std, color="r", alpha=0.25, label="Recall Score")
    # ax3.legend()

    m, l, u = analyze(precision, cl=2.5)
    a_plt_fill(ax4, x, m, l, u, "c", "Precision")

    # precision_std, precision_mean = get_mean_n_std(precision)
    # ax4.plot(x, precision_mean, "c-")
    # ax4.fill_between(
    #     x,
    #     precision_mean - precision_std,
    #     precision_mean + precision_std,
    #     color="c",
    #     alpha=0.25,
    #     label="Precision Score",
    # )
    # ax4.legend()

    m, l, u = analyze(frac, cl=2.5)
    a_plt_fill(ax5, x, m, l, u, "m", "Fraction of Anomalies detected")

    # frac_std, frac_mean = get_mean_n_std(frac)
    # ax5.plot(x, frac_mean, "m-")
    # ax5.fill_between(
    #     x,
    #     frac_mean - frac_std,
    #     frac_mean + frac_std,
    #     color="m",
    #     alpha=0.25,
    #     label="Fraction of anomalies detected Score",
    # )
    # ax5.legend()

    ax6.axis("off")

    sns.despine()
    plt.show()


def plot_all_single_metric(rws, mcc, recall, precision, frac, lbl, fig_size=(8, 8)):
    x = np.arange(len(rws[0]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=fig_size)
    fig.suptitle(lbl, fontsize=14)

    rws_std, rws_mean = get_mean_n_std(rws)
    ax1.plot(x, rws_mean, "g-")
    ax1.fill_between(x, rws_mean - rws_std, rws_mean + rws_std, color="g", alpha=0.25, label="RWS Score")
    ax1.legend()

    mcc_std, mcc_mean = get_mean_n_std(mcc)
    ax2.plot(x, rws_mean, "b-")
    ax2.fill_between(x, mcc_mean - mcc_std, mcc_mean + mcc_std, color="b", alpha=0.25, label="MCC Score")
    ax2.legend()

    recall_std, recall_mean = get_mean_n_std(recall)
    ax3.plot(x, recall_mean, "r-")
    ax3.fill_between(x, recall_mean - recall_std, recall_mean + recall_std, color="r", alpha=0.25, label="Recall Score")
    ax3.legend()

    precision_std, precision_mean = get_mean_n_std(precision)
    ax4.plot(x, precision_mean, "c-")
    ax4.fill_between(
        x,
        precision_mean - precision_std,
        precision_mean + precision_std,
        color="c",
        alpha=0.25,
        label="Precision Score",
    )
    ax4.legend()

    frac_std, frac_mean = get_mean_n_std(frac)
    ax5.plot(x, frac_mean, "m-")
    ax5.fill_between(
        x,
        frac_mean - frac_std,
        frac_mean + frac_std,
        color="m",
        alpha=0.25,
        label="Fraction of anomalies detected Score",
    )
    ax5.legend()

    ax6.axis("off")

    sns.despine()
    plt.show()


def plot_single(metric_list, metric_name, figsize=(8, 8)):

    n_rounds = len(metric_list)
    rounds = [x for x in range(1, n_rounds + 1)]

    plt.figure(figsize=figsize)
    plt.plot(rounds, metric_list)
    plt.xlabel("Rounds")
    plt.ylabel(metric_name)
    plt.show()


def plot_rws(iso_rws, latent_rws, ahunt_rws, figsize=(8, 8)):
    assert len(iso_rws) == len(latent_rws) == len(ahunt_rws)
    n_rounds = len(iso_rws)
    rounds = [x for x in range(1, n_rounds + 1)]
    plt.figure(figsize=figsize)
    plt.plot(rounds, iso_rws)
    plt.plot(rounds, latent_rws)
    plt.plot(rounds, ahunt_rws)
    plt.xlabel("Night")
    plt.ylabel("RWS")
    plt.legend(
        [
            "Isolation Forest",
            "Latent Space",
            "Ahunt",
        ],
        loc="upper left",
    )
    plt.show()


def plot_cm(truelabels, predictions, classes):
    cm = confusion_matrix(truelabels, predictions)
    tick_marks = np.arange(len(classes))

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt="g")
    plt.xlabel("Predicted ", fontsize=20)
    plt.ylabel("True ", fontsize=20)
    plt.show()


def plot_multiclass_auc_roc(classes, title, truelabels, predictions_probability):
    n_class = len(classes)
    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(truelabels, predictions_probability[:, i], pos_label=i)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label=f"{classes[0]} vs Rest")
    plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label=f"{classes[1]} vs Rest")
    plt.plot(fpr[2], tpr[2], linestyle="--", color="blue", label=f"{classes[2]} vs Rest")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")
    plt.legend(loc="best")
    plt.show()


def a_multiple_plt_fill_between(
    all_iso_rws,
    all_iso_mcc,
    all_iso_frac,
    all_iso_recall,
    all_iso_precision,
    all_latent_rws,
    all_latent_mcc,
    all_latent_frac,
    all_latent_recall,
    all_latent_precision,
    all_ahunt_rws,
    all_ahunt_mcc,
    all_ahunt_frac,
    all_ahunt_recall,
    all_ahunt_precision,
    font_size,
    figsize=(10, 12),
):
    #     plt.style.use("seaborn-dark-palette")

    x = np.arange(len(all_ahunt_rws[0]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.tight_layout(pad=4)

    fig.suptitle(
        f"Improvement of Ahunt after {len(all_ahunt_rws[0])} rounds of training",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.94)  # Aligns title properly in the presence of fig.tight_layout

    ### RWS

    m, l, u = analyze(all_iso_rws, cl=2.5)
    a_plt_fill(ax1, x, m, l, u, "y", "Isolation Forest")

    m, l, u = analyze(all_latent_rws, cl=2.5)
    a_plt_fill(ax1, x, m, l, u, "g", "Latent Learning")

    m, l, u = analyze(all_ahunt_rws, cl=2.5)
    a_plt_fill(ax1, x, m, l, u, "b", "Ahunt")

    ax1.set_xlabel("Rounds", fontsize=font_size)
    ax1.set_ylabel("RWS score", fontsize=font_size)

    ### MCC

    m, l, u = analyze(all_iso_mcc, cl=2.5)
    a_plt_fill(ax2, x, m, l, u, "y", "Isolation Forest")

    m, l, u = analyze(all_latent_mcc, cl=2.5)
    a_plt_fill(ax2, x, m, l, u, "g", "Latent Learning")

    m, l, u = analyze(all_ahunt_mcc, cl=2.5)
    a_plt_fill(ax2, x, m, l, u, "b", "Ahunt")

    ax2.set_xlabel("Rounds", fontsize=font_size)
    ax2.set_ylabel("MCC score", fontsize=font_size)

    ### FRAC

    m, l, u = analyze(all_iso_frac, cl=2.5)
    a_plt_fill(ax3, x, m, l, u, "y", "Isolation Forest")

    m, l, u = analyze(all_latent_frac, cl=2.5)
    a_plt_fill(ax3, x, m, l, u, "g", "Latent Learning")

    m, l, u = analyze(all_ahunt_frac, cl=2.5)
    a_plt_fill(ax3, x, m, l, u, "b", "Ahunt")

    ax3.set_xlabel("Rounds", fontsize=font_size)
    ax3.set_ylabel("Fraction of Anomalies", fontsize=font_size)

    ### RECALL

    m, l, u = analyze(all_iso_recall, cl=2.5)
    a_plt_fill(ax4, x, m, l, u, "y", "Isolation Forest")

    m, l, u = analyze(all_latent_recall, cl=2.5)
    a_plt_fill(ax4, x, m, l, u, "g", "Latent Learning")

    m, l, u = analyze(all_ahunt_recall, cl=2.5)
    a_plt_fill(ax4, x, m, l, u, "b", "Ahunt")

    ax4.set_xlabel("Rounds", fontsize=font_size)
    ax4.set_ylabel("Recall", fontsize=font_size)

    ### PRECISION

    m, l, u = analyze(all_iso_precision, cl=2.5)
    a_plt_fill(ax5, x, m, l, u, "y", "Isolation Forest")

    m, l, u = analyze(all_latent_precision, cl=2.5)
    a_plt_fill(ax5, x, m, l, u, "g", "Latent Learning")

    m, l, u = analyze(all_ahunt_precision, cl=2.5)
    a_plt_fill(ax5, x, m, l, u, "b", "Ahunt")

    ax5.set_xlabel("Rounds", fontsize=font_size)
    ax5.set_ylabel("Precision", fontsize=font_size)

    ax6.axis("off")

    sns.despine()

    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
