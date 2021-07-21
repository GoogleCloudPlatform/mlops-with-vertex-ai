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
"""Train and evaluate the model."""

import logging
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras


from src.model_training import data, model


def train(
    train_data_dir,
    eval_data_dir,
    tft_output_dir,
    hyperparams,
    log_dir,
    base_model_dir=None,
):

    logging.info(f"Loading tft output from {tft_output_dir}")
    tft_output = tft.TFTransformOutput(tft_output_dir)
    transformed_feature_spec = tft_output.transformed_feature_spec()

    train_dataset = data.get_dataset(
        train_data_dir,
        transformed_feature_spec,
        hyperparams["batch_size"],
    )

    eval_dataset = data.get_dataset(
        eval_data_dir,
        transformed_feature_spec,
        hyperparams["batch_size"],
    )

    optimizer = keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    classifier = model.create_binary_classifier(tft_output, hyperparams)
    if base_model_dir:
        try:
            classifier = keras.load_model(base_model_dir)
        except:
            pass

    classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logging.info("Model training started...")
    classifier.fit(
        train_dataset,
        epochs=hyperparams["num_epochs"],
        validation_data=eval_dataset,
        callbacks=[early_stopping, tensorboard_callback],
    )
    logging.info("Model training completed.")

    return classifier


def evaluate(model, data_dir, raw_schema_location, tft_output_dir, hyperparams):
    logging.info(f"Loading raw schema from {raw_schema_location}")

    logging.info(f"Loading tft output from {tft_output_dir}")
    tft_output = tft.TFTransformOutput(tft_output_dir)
    transformed_feature_spec = tft_output.transformed_feature_spec()

    logging.info("Model evaluation started...")
    eval_dataset = data.get_dataset(
        data_dir,
        transformed_feature_spec,
        hyperparams["batch_size"],
    )

    evaluation_metrics = model.evaluate(eval_dataset)
    logging.info("Model evaluation completed.")

    return evaluation_metrics
