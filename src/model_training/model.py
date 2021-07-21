# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A DNN keras classification model."""

import tensorflow as tf
from tensorflow import keras

from src.common import features


def create_model_inputs():
    inputs = {}
    for feature_name in features.FEATURE_NAMES:
        name = features.transformed_name(feature_name)
        if feature_name in features.NUMERICAL_FEATURE_NAMES:
            inputs[name] = keras.layers.Input(name=name, shape=[], dtype=tf.float32)
        elif feature_name in features.categorical_feature_names():
            inputs[name] = keras.layers.Input(name=name, shape=[], dtype=tf.int64)
        else:
            pass
    return inputs


def _create_binary_classifier(feature_vocab_sizes, hyperparams):
    input_layers = create_model_inputs()

    layers = []
    for key in input_layers:
        feature_name = features.original_name(key)
        if feature_name in features.EMBEDDING_CATEGORICAL_FEATURES:
            vocab_size = feature_vocab_sizes[feature_name]
            embedding_size = features.EMBEDDING_CATEGORICAL_FEATURES[feature_name]
            embedding_output = keras.layers.Embedding(
                input_dim=vocab_size + 1,
                output_dim=embedding_size,
                name=f"{key}_embedding",
            )(input_layers[key])
            layers.append(embedding_output)
        elif feature_name in features.ONEHOT_CATEGORICAL_FEATURE_NAMES:
            vocab_size = feature_vocab_sizes[feature_name]
            onehot_layer = keras.layers.experimental.preprocessing.CategoryEncoding(
                max_tokens=vocab_size,
                output_mode="binary",
                name=f"{key}_onehot",
            )(input_layers[key])
            layers.append(onehot_layer)
        elif feature_name in features.NUMERICAL_FEATURE_NAMES:
            numeric_layer = tf.expand_dims(input_layers[key], -1)
            layers.append(numeric_layer)
        else:
            pass

    joined = keras.layers.Concatenate(name="combines_inputs")(layers)
    feedforward_output = keras.Sequential(
        [
            keras.layers.Dense(units, activation="relu")
            for units in hyperparams["hidden_units"]
        ],
        name="feedforward_network",
    )(joined)
    logits = keras.layers.Dense(units=1, name="logits")(feedforward_output)

    model = keras.Model(inputs=input_layers, outputs=[logits])
    return model


def create_binary_classifier(tft_output, hyperparams):
    feature_vocab_sizes = dict()
    for feature_name in features.categorical_feature_names():
        feature_vocab_sizes[feature_name] = tft_output.vocabulary_size_by_name(
            feature_name
        )

    return _create_binary_classifier(feature_vocab_sizes, hyperparams)
