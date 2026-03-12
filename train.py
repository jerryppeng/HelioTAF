import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from data_loader import data_loader_image_with_features, data_loader_image_only
from models.finetune_10_pvnet_with_all_ra_lstm_t_timesteps import build_cnn_lstm_with_pretrained_pvnet
from models.finetune_aff_10_pvnet_with_all_ra_lstm_t_timesteps import build_cnn_lstm_with_pretrained_pvnet_and_taf
from models.pvnet_with_lstm_t_timesteps import (
    build_cnn_lstm_with_pretrained_pvnet_only,
    load_pvnet_feature_extractor,
)
from utils import cv_split, day_block_shuffle


class Config:
    project_root = Path(__file__).resolve().parent
    data_folder = Path(os.getenv("PVNET_DATA_DIR", project_root / "data"))
    output_root = Path(os.getenv("PVNET_OUTPUT_DIR", project_root / "outputs"))
    pretrained_pvnet_path = Path(
        os.getenv(
            "PRETRAINED_PVNET_PATH",
            project_root / "checkpoints" / "pvnet" / "best_model.h5",
        )
    )

    specified_base_path = str(output_root)
    model_name = "Fusion-PVnet-for-15"
    data_path = str(data_folder / "train_2019_for_15.h5")
    output_folder = str(output_root / "model_output" / model_name)

    num_filters = 12
    num_epochs = 100
    num_fold = 5
    batch_size = 64
    timesteps = 8
    initial_learning_rate = 2e-3

    USE_FEATURE_LOADER = True  # True: image + numeric features, False: image only.
    MODEL_TYPE = "PVNET"  # Options: "TAF", "PVNET", "PVNET_ONLY".


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


create_directory(Config.output_folder)


def select_data_loader(data_path, indices, batch_size, timesteps, use_features=False):
    if use_features:
        return data_loader_image_with_features(data_path, indices, batch_size=batch_size, timesteps=timesteps)
    return data_loader_image_only(data_path, indices, batch_size=batch_size, timesteps=timesteps)


def select_model(timesteps):
    pvnet_feature_extractor = load_pvnet_feature_extractor(Config.pretrained_pvnet_path)
    if Config.MODEL_TYPE == "TAF":
        return build_cnn_lstm_with_pretrained_pvnet_and_taf(pvnet_feature_extractor, timesteps=timesteps)
    if Config.MODEL_TYPE == "PVNET":
        return build_cnn_lstm_with_pretrained_pvnet(pvnet_feature_extractor, timesteps=timesteps)
    if Config.MODEL_TYPE == "PVNET_ONLY":
        return build_cnn_lstm_with_pretrained_pvnet_only(pvnet_feature_extractor, timesteps=timesteps)
    raise ValueError('Invalid MODEL_TYPE. Choose from "TAF", "PVNET", or "PVNET_ONLY".')


def setup_gpu():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print("Available CPUs:", os.cpu_count())

setup_gpu()


def train_model():
    trainval_timestamps = np.load(str(Config.data_folder / "sampled_times_full_train_time_2019.npy"), allow_pickle=True)
    indices_dayblock_shuffled = day_block_shuffle(trainval_timestamps)

    train_loss_hist = []
    val_loss_hist = []

    for i in range(Config.num_fold):
        keras.backend.clear_session()
        model = select_model(Config.timesteps)
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(Config.initial_learning_rate), loss="mse")

        def lr_time_based_decay(epoch, lr):
            _ = lr
            drop_rate = 0.5
            epochs_drop = 10.0
            return Config.initial_learning_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))

        save_directory = os.path.join(Config.output_folder, f"repetition_{i + 1}")
        create_directory(save_directory)

        indices_train, indices_val = cv_split(indices_dayblock_shuffled, i, Config.num_fold)
        ds_train_batched = select_data_loader(Config.data_path, indices_train, Config.batch_size, Config.timesteps, Config.USE_FEATURE_LOADER)
        ds_val_batched = select_data_loader(Config.data_path, indices_val, Config.batch_size, Config.timesteps, Config.USE_FEATURE_LOADER)

        earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(save_directory, f"best_model_repetition_{i + 1}.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        lr_schedule = keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

        history = model.fit(
            ds_train_batched,
            epochs=Config.num_epochs,
            steps_per_epoch=len(indices_train) // Config.batch_size + 1,
            verbose=1,
            callbacks=[earlystop, checkpoint, lr_schedule],
            validation_data=ds_val_batched,
            validation_steps=len(indices_val) // Config.batch_size + 1,
        )

        train_loss_hist.append(history.history["loss"])
        val_loss_hist.append(history.history["val_loss"])
        np.save(os.path.join(Config.output_folder, f"train_loss_hist_fold_{i + 1}.npy"), history.history["loss"])
        np.save(os.path.join(Config.output_folder, f"val_loss_hist_fold_{i + 1}.npy"), history.history["val_loss"])

        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="validation")
        plt.legend()
        plt.savefig(os.path.join(Config.output_folder, f"loss_plot_fold_{i + 1}.png"))
        plt.clf()

    summarize_results(train_loss_hist, val_loss_hist)


def summarize_results(train_loss_hist, val_loss_hist):
    best_train_loss_MSE = np.zeros(Config.num_fold)
    best_val_loss_MSE = np.zeros(Config.num_fold)

    for i in range(Config.num_fold):
        best_val_loss_MSE[i] = np.min(val_loss_hist[i])
        idx = np.argmin(val_loss_hist[i])
        best_train_loss_MSE[i] = train_loss_hist[i][idx]
        print(
            f"Model {i + 1}  -- train loss: {np.sqrt(best_train_loss_MSE[i]):.2f}, "
            f"validation loss: {np.sqrt(best_val_loss_MSE[i]):.2f} (RMSE)"
        )

    print(f"The mean train loss (RMSE) for all models is {np.mean(np.sqrt(best_train_loss_MSE)):.2f}")
    print(f"The mean validation loss (RMSE) for all models is {np.mean(np.sqrt(best_val_loss_MSE)):.2f}")


if __name__ == "__main__":
    train_model()
