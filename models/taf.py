import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, GlobalAveragePooling1D


class TAF(tf.keras.layers.Layer):
    def __init__(self, feature_dim=64, reduction_ratio=4, **kwargs):
        super(TAF, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio
        inter_features = feature_dim // reduction_ratio

        # Local Feature Extraction - Extract features locally across time using dense layers and batch normalization
        self.local_feature_fc1 = Dense(inter_features, activation='relu')  # First dense layer for local feature extraction
        self.local_feature_bn1 = BatchNormalization()  # Batch normalization to stabilize training
        self.local_feature_fc2 = Dense(feature_dim, activation='relu')  # Second dense layer to project back to original feature dimension
        self.local_feature_bn2 = BatchNormalization()  # Batch normalization to stabilize training

        # Global Feature Extraction - Extract global context using pooling and dense layers
        self.global_pooling = GlobalAveragePooling1D()  # Global average pooling to summarize features across time steps
        self.global_feature_fc1 = Dense(inter_features, activation='relu')  # First dense layer for global feature extraction
        self.global_feature_bn1 = BatchNormalization()  # Batch normalization to stabilize training
        self.global_feature_fc2 = Dense(feature_dim, activation='relu')  # Second dense layer to project back to original feature dimension
        self.global_feature_bn2 = BatchNormalization()  # Batch normalization to stabilize training

        # Sigmoid activation to generate attention weights
        self.sigmoid_activation = Activation('sigmoid')

    def call(self, inputs_1, inputs_2):
        # Input shapes for inputs_1, inputs_2: (batch_size, timesteps, feature_dim)
        # Fuse the inputs by element-wise addition
        fused_inputs = Add()([inputs_1, inputs_2])

        # Local Feature Extraction - Extract temporal local features using dense layers
        local_feature_output = self.local_feature_fc1(fused_inputs)  # Apply first dense layer
        local_feature_output = self.local_feature_bn1(local_feature_output)  # Apply batch normalization
        local_feature_output = self.local_feature_fc2(local_feature_output)  # Apply second dense layer
        local_feature_output = self.local_feature_bn2(local_feature_output)  # Apply batch normalization

        # Global Feature Extraction - Extract temporal global features using pooling and dense layers
        global_feature_output = self.global_pooling(fused_inputs)  # Apply global average pooling
        global_feature_output = self.global_feature_fc1(global_feature_output)  # Apply first dense layer
        global_feature_output = self.global_feature_bn1(global_feature_output)  # Apply batch normalization
        global_feature_output = self.global_feature_fc2(global_feature_output)  # Apply second dense layer
        global_feature_output = self.global_feature_bn2(global_feature_output)  # Apply batch normalization

        # Reshape global feature output for fusion
        global_feature_output = tf.keras.layers.Reshape((1, -1))(global_feature_output)  # Reshape to match local feature dimensions
        combined_feature_output = Add()([local_feature_output, global_feature_output])  # Combine local and global features
        attention_weights = self.sigmoid_activation(combined_feature_output)  # Generate attention weights using sigmoid activation

        # Weighted fusion of inputs
        output = inputs_1 * attention_weights + inputs_2 * (1 - attention_weights)  # Apply attention weights to fuse inputs

        return output

    def get_config(self):
        # Return the configuration of the layer for serialization
        config = super(TAF, self).get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "reduction_ratio": self.reduction_ratio,
        })
        return config
