from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model


def build_cnn_lstm_with_pretrained_pvnet(
    pvnet_feature_extractor,
    input_shape=(64, 64, 3),
    num_features=17,
    timesteps=8,
):
    """Build a dual-branch PV forecasting model with image and numeric features."""
    image_sequence_input = Input(shape=(timesteps,) + input_shape, name="image_input")
    encoded_sequence = TimeDistributed(pvnet_feature_extractor, name="pvnet_feature_extractor")(image_sequence_input)
    encoded_sequence = TimeDistributed(BatchNormalization(), name="batch_norm_pvnet")(encoded_sequence)

    features_input = Input(shape=(timesteps, num_features), name="features_input")
    numeric_branch = TimeDistributed(Dense(128, activation="relu"), name="dense_128_features")(features_input)
    numeric_branch = TimeDistributed(BatchNormalization(), name="batch_norm_128_features")(numeric_branch)
    numeric_branch = TimeDistributed(Dense(64, activation="relu"), name="dense_64_features")(numeric_branch)
    numeric_branch = TimeDistributed(BatchNormalization(), name="batch_norm_64_features")(numeric_branch)
    numeric_branch = TimeDistributed(Dense(32, activation="relu"), name="dense_32_features")(numeric_branch)

    fused_features = Concatenate(axis=-1, name="concat_fusion")([encoded_sequence, numeric_branch])
    lstm_output = LSTM(64, activation="tanh", name="lstm_layer")(fused_features)

    dense_output = Dense(32, activation="relu", name="dense_fc")(lstm_output)
    dense_output = BatchNormalization(name="batch_norm_fc")(dense_output)
    prediction = Dense(1, activation="linear", name="output")(dense_output)

    return Model(inputs=[image_sequence_input, features_input], outputs=prediction, name="cnn_lstm_with_pvnet")
