import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.taf import TAF
from utils import mask_background


class Config:
    num_fold = 5
    batch_size = 100
    project_root = Path(__file__).resolve().parent
    data_folder = Path(os.getenv("PVNET_DATA_DIR", project_root / "data"))
    output_root = Path(os.getenv("PVNET_OUTPUT_DIR", project_root / "outputs"))

    specified_base_path = str(output_root)
    model_name = "Fusion-PVnet-for-15"
    output_folder = str(output_root / "model_output" / model_name)
    test_data_path = str(data_folder / "test_2018_sampled_for_15.h5")
    test_timestamps_path = str(data_folder / "test_time_2018.npy")
    use_numeric_features = True  # True: use (images, numeric_features); False: use images only.


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


create_directory(Config.output_folder)
print("Output folder:", Config.output_folder)


def load_timestamps(path):
    return np.load(path, allow_pickle=True)


times_test = load_timestamps(Config.test_timestamps_path)
print(times_test.shape)

if not os.path.isfile(Config.test_data_path):
    raise FileNotFoundError("Test data file does not exist. Please check the path.")
print("Test data file exists. Loading data...")


def load_test_data(path, batch_size):
    with h5py.File(path, "r") as f:
        images_log = f["extracted_images_log"][:]
        pv_log = f["pv_values"][:]
        features = [
            f["temperature_2m (°C)"][:],
            f["relative_humidity_2m (%)"][:],
            f["dew_point_2m (°C)"][:],
            f["apparent_temperature (°C)"][:],
            f["cloud_cover (%)"][:],
            f["cloud_cover_low (%)"][:],
            f["wind_direction_10m (°)"][:],
            f["wind_direction_100m (°)"][:],
            f["wind_gusts_10m (km_h)"][:],
            f["wind_speed_100m (km_h)"][:],
            f["weather_code (wmo code)"][:],
            f["surface_pressure (hPa)"][:],
            f["cloud_cover_mid (%)"][:],
            f["cloud_cover_high (%)"][:],
            f["et0_fao_evapotranspiration (mm)"][:],
            f["vapour_pressure_deficit (kPa)"][:],
            f["wind_speed_10m (km_h)"][:],
        ]
        numeric_features = np.stack(features, axis=-1)

        images_data_test = [
            mask_background(tf.image.convert_image_dtype(images_log[start:end], tf.float32).numpy())
            for start in range(0, images_log.shape[0], batch_size)
            for end in [min(start + batch_size, images_log.shape[0])]
        ]

        images_data_test = np.concatenate(images_data_test, axis=0)
        pv_log_test = tf.convert_to_tensor(pv_log, dtype=tf.float32).numpy()
        return images_data_test, pv_log_test, numeric_features


images_data_test, pv_log_test, numeric_features = load_test_data(Config.test_data_path, Config.batch_size)
print("images_data_test.shape:", images_data_test.shape)
print("pv_log_test.shape:", pv_log_test.shape)
print("numeric_features.shape:", numeric_features.shape)


def evaluate_and_predict_model(num_fold, images_data_test, pv_log_test, numeric_features):
    losses = np.zeros((num_fold, len(times_test)))
    predictions = np.zeros((num_fold, len(times_test)))

    for i in range(num_fold):
        print(f"Loading repetition {i + 1} model ...")
        model_path = os.path.join(Config.output_folder, f"repetition_{i + 1}", f"best_model_repetition_{i + 1}.h5")
        model = keras.models.load_model(model_path, custom_objects={"TAF": TAF})

        test_inputs = (images_data_test, numeric_features) if Config.use_numeric_features else images_data_test

        with tf.device("/CPU:0"):
            print(f"Evaluating performance for model {i + 1}")
            losses[i] = model.evaluate(x=test_inputs, y=pv_log_test, batch_size=64, verbose=1)
            print(f"Generating predictions for model {i + 1}")
            predictions[i] = np.squeeze(model.predict(test_inputs, batch_size=64, verbose=1))

    return losses, predictions


loss, prediction = evaluate_and_predict_model(Config.num_fold, images_data_test, pv_log_test, numeric_features)
np.save(os.path.join(Config.output_folder, "test_predictions_validation.npy"), prediction)

# Use all trained sub-models for a single ensemble prediction.
prediction_ensemble = np.mean(prediction, axis=0)
loss_ensemble = np.sqrt(np.mean((prediction_ensemble - pv_log_test) ** 2))
r2_ensemble = 1.0 - (np.sum((pv_log_test - prediction_ensemble) ** 2) / np.sum((pv_log_test - np.mean(pv_log_test)) ** 2))
print(f"The test set RMSE is {loss_ensemble:.3f} for the ensemble model")
print(f"The test set R^2 is {r2_ensemble:.4f} for the ensemble model")

with open(os.path.join(Config.output_folder, "output_results.txt"), "a") as f:
    f.write(f"All-model ensemble RMSE: {loss_ensemble:.3f} kW\n")
    f.write(f"All-model ensemble R^2: {r2_ensemble:.4f}\n")
