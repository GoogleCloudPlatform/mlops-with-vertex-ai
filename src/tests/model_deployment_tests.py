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
"""Test an uploaded model to Vertex AI."""

import os
import logging
import tensorflow as tf

test_instance = {
    "dropoff_grid": ["POINT(-87.6 41.9)"],
    "euclidean": [2064.2696],
    "loc_cross": [""],
    "payment_type": ["Credit Card"],
    "pickup_grid": ["POINT(-87.6 41.9)"],
    "trip_miles": [1.37],
    "trip_day": [12],
    "trip_hour": [16],
    "trip_month": [2],
    "trip_day_of_week": [4],
    "trip_seconds": [555],
}

SERVING_DEFAULT_SIGNATURE_NAME = "serving_default"

from google.cloud import aiplatform as vertex_ai


def test_model_artifact():

    feature_types = {
        "dropoff_grid": tf.dtypes.string,
        "euclidean": tf.dtypes.float32,
        "loc_cross": tf.dtypes.string,
        "payment_type": tf.dtypes.string,
        "pickup_grid": tf.dtypes.string,
        "trip_miles": tf.dtypes.float32,
        "trip_day": tf.dtypes.int64,
        "trip_hour": tf.dtypes.int64,
        "trip_month": tf.dtypes.int64,
        "trip_day_of_week": tf.dtypes.int64,
        "trip_seconds": tf.dtypes.int64,
    }

    new_test_instance = dict()
    for key in test_instance:
        new_test_instance[key] = tf.constant(
            [test_instance[key]], dtype=feature_types[key]
        )

    print(new_test_instance)

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"

    vertex_ai.init(project=project, location=region,)

    models = vertex_ai.Model.list(
        filter=f'display_name={model_display_name}',
        order_by="update_time"
    )

    assert (
        models
    ), f"No model with display name {model_display_name} exists!"

    model = models[-1]
    artifact_uri = model.gca_resource.artifact_uri
    logging.info(f"Model artifact uri:{artifact_uri}")
    assert tf.io.gfile.exists(
        artifact_uri
    ), f"Model artifact uri {artifact_uri} does not exist!"

    saved_model = tf.saved_model.load(artifact_uri)
    logging.info("Model loaded successfully.")

    assert (
        SERVING_DEFAULT_SIGNATURE_NAME in saved_model.signatures
    ), f"{SERVING_DEFAULT_SIGNATURE_NAME} not in model signatures!"

    prediction_fn = saved_model.signatures["serving_default"]
    predictions = prediction_fn(**new_test_instance)
    logging.info("Model produced predictions.")

    keys = ["classes", "scores"]
    for key in keys:
        assert key in predictions, f"{key} in prediction outputs!"

    assert predictions["classes"].shape == (
        1,
        2,
    ), f"Invalid output classes shape: {predictions['classes'].shape}!"
    assert predictions["scores"].shape == (
        1,
        2,
    ), f"Invalid output scores shape: {predictions['scores'].shape}!"
    logging.info(f"Prediction output: {predictions}")


def test_model_endpoint():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    endpoint_display_name = os.getenv("ENDPOINT_DISPLAY_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"
    assert endpoint_display_name, "Environment variable ENDPOINT_DISPLAY_NAME is None!"

    endpoints = vertex_ai.Endpoint.list(
        filter=f'display_name={endpoint_display_name}',
        order_by="update_time"
    )
    assert (
        endpoints
    ), f"Endpoint with display name {endpoint_display_name} does not exist! in region {region}"

    endpoint = endpoints[-1]
    logging.info(f"Calling endpoint: {endpoint}.")

    prediction = endpoint.predict([test_instance]).predictions[0]

    keys = ["classes", "scores"]
    for key in keys:
        assert key in prediction, f"{key} in prediction outputs!"

    assert (
        len(prediction["classes"]) == 2
    ), f"Invalid number of output classes: {len(prediction['classes'])}!"
    assert (
        len(prediction["scores"]) == 2
    ), f"Invalid number output scores: {len(prediction['scores'])}!"

    logging.info(f"Prediction output: {prediction}")
