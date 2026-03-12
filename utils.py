import datetime
import itertools

import numpy as np
from matplotlib import pyplot as plt


def day_block_shuffle(times_trainval):
    """Shuffle indices by day blocks so temporal neighbors stay grouped."""
    times_trainval = np.array(times_trainval, dtype="datetime64[s]")
    dates_trainval = np.zeros_like(times_trainval, dtype=object)

    for i in range(len(times_trainval)):
        dates_trainval[i] = datetime.datetime.utcfromtimestamp(times_trainval[i].astype(int)).date()

    unique_dates = np.unique(dates_trainval)
    blocks = []
    for date in unique_dates:
        blocks.append(np.where(dates_trainval == date)[0])

    np.random.seed(1)
    np.random.shuffle(blocks)
    return np.asarray(list(itertools.chain.from_iterable(blocks)))


def cv_split(split_data, fold_index, num_fold):
    """Split shuffled indices into train/validation folds."""
    num_samples = len(split_data)
    indices = np.arange(num_samples)

    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[int(fold_index / num_fold * num_samples) : int((fold_index + 1) / num_fold * num_samples)] = True
    val_indices = indices[val_mask]
    train_indices = indices[np.logical_not(val_mask)]

    np.random.seed(fold_index)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    data_train = split_data[train_indices]
    print("Train split info:", data_train.shape)
    data_val = split_data[val_indices]
    print("Validation split info:", data_val.shape)
    return data_train, data_val


def mask_background(img):
    """Set non-sky pixels (outside circular fisheye region) to zero."""
    mask = np.ones((64, 64, 3), dtype=bool)
    for i in range(64):
        for j in range(64):
            if (i - 30) ** 2 + (j - 30) ** 2 >= 31**2:
                mask[i, j, :] = 0
    return img * mask


def compute_winkler_score(prob_prediction, observation):
    """Compute Winkler interval score for a single observation."""
    alpha = 0.1
    lb = np.percentile(prob_prediction, 5, axis=0)
    ub = np.percentile(prob_prediction, 95, axis=0)
    delta = ub - lb

    if observation < lb:
        return delta + 2 * (lb - observation) / alpha
    if observation > ub:
        return delta + 2 * (observation - ub) / alpha
    return delta


def plot_lr(history):
    """Plot learning-rate history from a Keras History object."""
    learning_rate = history.history["lr"]
    epochs = range(1, len(learning_rate) + 1)
    plt.plot(epochs, learning_rate)
    plt.title("Learning rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.show()
