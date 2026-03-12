import h5py
import numpy as np
import tensorflow as tf

from utils import mask_background


def data_loader_image_with_features(hdf5_data_path, sample_idx, batch_size=128, timesteps=9):
    """Build a tf.data pipeline for image sequences + numeric weather features."""

    def mapping_func_py(hdf5_data_path, sample_idx):
        hdf5_data_path = hdf5_data_path.numpy().decode()
        sample_idx = sorted(sample_idx.numpy())

        with h5py.File(hdf5_data_path, "r") as f:
            images_log = tf.image.convert_image_dtype(f["extracted_images_log"][sample_idx], tf.float32)
            images_log = mask_background(images_log)
            pv_log = tf.convert_to_tensor(f["pv_values"][sample_idx], dtype=tf.float32)

            numeric_features = np.stack(
                [
                    f["temperature_2m (°C)"][sample_idx],
                    f["relative_humidity_2m (%)"][sample_idx],
                    f["dew_point_2m (°C)"][sample_idx],
                    f["apparent_temperature (°C)"][sample_idx],
                    f["cloud_cover (%)"][sample_idx],
                    f["cloud_cover_low (%)"][sample_idx],
                    f["wind_direction_10m (°)"][sample_idx],
                    f["wind_direction_100m (°)"][sample_idx],
                    f["wind_gusts_10m (km_h)"][sample_idx],
                    f["wind_speed_100m (km_h)"][sample_idx],
                    f["weather_code (wmo code)"][sample_idx],
                    f["surface_pressure (hPa)"][sample_idx],
                    f["cloud_cover_mid (%)"][sample_idx],
                    f["cloud_cover_high (%)"][sample_idx],
                    f["et0_fao_evapotranspiration (mm)"][sample_idx],
                    f["vapour_pressure_deficit (kPa)"][sample_idx],
                    f["wind_speed_10m (km_h)"][sample_idx],
                ],
                axis=-1,
            )
            numeric_features = tf.convert_to_tensor(numeric_features, dtype=tf.float32)

        return images_log, numeric_features, pv_log

    def mapping_func_tf(hdf5_data_path, sample_idx):
        images_log, numeric_features, pv_log = tf.py_function(
            func=mapping_func_py,
            inp=[hdf5_data_path, sample_idx],
            Tout=(tf.float32, tf.float32, tf.float32),
        )
        images_log.set_shape([None, timesteps, 64, 64, 3])
        numeric_features.set_shape([None, timesteps, 17])
        pv_log.set_shape([None])
        return (images_log, numeric_features), pv_log

    sample_idx = tf.convert_to_tensor(sample_idx, dtype=tf.int64)
    idx_ds = tf.data.Dataset.from_tensor_slices(sample_idx)
    idx_ds = idx_ds.shuffle(buffer_size=len(sample_idx), seed=0)
    idx_ds = idx_ds.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return idx_ds.map(lambda x: mapping_func_tf(hdf5_data_path, x), num_parallel_calls=tf.data.experimental.AUTOTUNE)


def data_loader_image_only(hdf5_data_path, sample_idx, batch_size=256, timesteps=24):
    """Build a tf.data pipeline for image sequences only."""

    def mapping_func_py(hdf5_data_path, sample_idx):
        hdf5_data_path = hdf5_data_path.numpy().decode()
        sample_idx = sorted(sample_idx.numpy())

        with h5py.File(hdf5_data_path, "r") as f:
            images_log = f["extracted_images_log"][sample_idx]
            pv_log = f["pv_values"][sample_idx]

            images_log = tf.image.convert_image_dtype(images_log, tf.float32)
            images_log = mask_background(images_log)
            pv_log = tf.convert_to_tensor(pv_log, dtype=tf.float32)

            return images_log, pv_log

    def mapping_func_tf(hdf5_data_path, sample_idx):
        images_log, pv_log = tf.py_function(
            func=mapping_func_py,
            inp=[hdf5_data_path, sample_idx],
            Tout=(tf.float32, tf.float32),
        )
        images_log.set_shape([None, timesteps, 64, 64, 3])
        pv_log.set_shape([None])
        return images_log, pv_log

    sample_idx = tf.convert_to_tensor(sample_idx, dtype=tf.int64)
    idx_ds = tf.data.Dataset.from_tensor_slices(sample_idx)
    idx_ds = idx_ds.shuffle(buffer_size=len(sample_idx), seed=0)
    idx_ds = idx_ds.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    return idx_ds.map(lambda x: mapping_func_tf(hdf5_data_path, x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
