from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model

from models.taf import TAF


def build_cnn_lstm_with_pretrained_pvnet_and_taf(
    pvnet_feature_extractor,
    input_shape=(64, 64, 3),
    num_features=17,
    timesteps=24,
):
    """Build a PVNet model with TAF temporal fusion for image and numeric branches."""
    image_sequence_input = Input(shape=(timesteps,) + input_shape, name="image_input_sequence")
    encoded_sequence = TimeDistributed(pvnet_feature_extractor, name="time_distributed_pvnet")(image_sequence_input)
    encoded_sequence = TimeDistributed(BatchNormalization(), name="batch_norm_pvnet")(encoded_sequence)

    features_input = Input(shape=(timesteps, num_features), name="features_input")
    numeric_branch = TimeDistributed(Dense(128, activation="relu"), name="dense_128_features")(features_input)
    numeric_branch = TimeDistributed(BatchNormalization(), name="batch_norm_128_features")(numeric_branch)
    numeric_branch = TimeDistributed(Dense(64, activation="relu"), name="dense_64_features")(numeric_branch)
    numeric_branch = TimeDistributed(BatchNormalization(), name="batch_norm_64_features")(numeric_branch)
    numeric_branch = TimeDistributed(Dense(32, activation="relu"), name="dense_32_features")(numeric_branch)

    encoded_sequence_lstm = LSTM(64, return_sequences=True, dropout=0.2, name="lstm_encoded_sequence")(encoded_sequence)
    numeric_branch_lstm = LSTM(64, return_sequences=True, dropout=0.2, name="lstm_numeric_branch")(numeric_branch)

    taf_output = TAF(feature_dim=64, name="taf_fusion")(encoded_sequence_lstm, numeric_branch_lstm)

    lstm_output = LSTM(128, activation="tanh", return_sequences=True, dropout=0.3, name="lstm_128")(taf_output)
    lstm_output = LSTM(64, activation="tanh", dropout=0.3, name="lstm_64")(lstm_output)

    dense_output = Dense(64, activation="relu", name="dense_64")(lstm_output)
    dense_output = BatchNormalization(name="batch_norm_fc")(dense_output)
    dense_output = Dropout(0.4, name="dropout_fc")(dense_output)
    prediction = Dense(1, activation="linear", name="output")(dense_output)

    return Model(inputs=[image_sequence_input, features_input], outputs=prediction, name="cnn_lstm_pvnet_taf_model")
