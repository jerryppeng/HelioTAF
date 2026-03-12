from pathlib import Path

from tensorflow.keras.layers import BatchNormalization, Dense, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model, load_model


def load_pvnet_feature_extractor(pretrained_model_path):
    """Load a pretrained PVNet checkpoint and return its feature extractor."""
    pretrained_model_path = Path(pretrained_model_path)
    if not pretrained_model_path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_model_path}. "
            "Set PRETRAINED_PVNET_PATH in your config/env to a valid .h5 checkpoint."
        )

    pvnet_model = load_model(str(pretrained_model_path))
    feature_extractor = Model(inputs=pvnet_model.input, outputs=pvnet_model.layers[-2].output)
    feature_extractor.trainable = True
    return feature_extractor


def build_cnn_lstm_with_pretrained_pvnet_only(
    pvnet_feature_extractor,
    input_shape=(64, 64, 3),
    timesteps=7,
):
    """Build an image-only PVNet + LSTM regression model."""
    image_sequence_input = Input(shape=(timesteps,) + input_shape, name="image_input")
    encoded_sequence = TimeDistributed(pvnet_feature_extractor, name="pvnet_feature_extractor")(image_sequence_input)
    encoded_sequence = BatchNormalization(name="batch_norm_pvnet")(encoded_sequence)

    lstm_output = LSTM(64, activation="tanh", name="lstm_layer")(encoded_sequence)
    lstm_output = BatchNormalization(name="batch_norm_lstm")(lstm_output)

    dense_output = Dense(32, activation="tanh", name="dense_fc")(lstm_output)
    dense_output = BatchNormalization(name="batch_norm_fc")(dense_output)
    prediction = Dense(1, activation="linear", name="output_layer")(dense_output)

    return Model(inputs=image_sequence_input, outputs=prediction, name="cnn_lstm_with_pvnet")
