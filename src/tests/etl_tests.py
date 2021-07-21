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
"""Test data processing."""

import sys
import os
import logging
import tensorflow_transform as tft
import tensorflow as tf
from tensorflow.io import FixedLenFeature

from src.preprocessing import etl
from src.comm import datasource_utils

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

OUTPUT_DIR = "test_etl_output_dir"
ML_USE = "UNASSIGNED"
LIMIT = 100

EXPECTED_FEATURE_SPEC = {
    "dropoff_grid_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "euclidean_xf": FixedLenFeature(shape=[], dtype=tf.float32, default_value=None),
    "loc_cross_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "payment_type_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "pickup_grid_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "tip_bin": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "trip_day_of_week_xf": FixedLenFeature(
        shape=[], dtype=tf.int64, default_value=None
    ),
    "trip_day_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "trip_hour_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "trip_miles_xf": FixedLenFeature(shape=[], dtype=tf.float32, default_value=None),
    "trip_month_xf": FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    "trip_seconds_xf": FixedLenFeature(shape=[], dtype=tf.float32, default_value=None),
}


def test_transform_pipeline():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    bucket = os.getenv("BUCKET")
    dataset_display_name = os.getenv("DATASET_DISPLAY_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert bucket, "Environment variable BUCKET is None!"
    assert dataset_display_name, "Environment variable DATASET_DISPLAY_NAME is None!"

    os.mkdir(OUTPUT_DIR)

    exported_data_dir = os.path.join(OUTPUT_DIR, "exported_data")
    transformed_data_dir = os.path.join(OUTPUT_DIR, "transformed_data")
    transform_artifacts_dir = os.path.join(OUTPUT_DIR, "transform_artifacts")
    temporary_dir = os.path.join(OUTPUT_DIR, "tmp")

    raw_data_query = datasource_utils.get_training_source_query(
        project=project,
        region=region,
        dataset_display_name=dataset_display_name,
        ml_use=ML_USE,
        limit=LIMIT,
    )

    args = {
        "runner": "DirectRunner",
        "raw_data_query": raw_data_query,
        "write_raw_data": False,
        "exported_data_prefix": exported_data_dir,
        "transformed_data_prefix": transformed_data_dir,
        "transform_artefact_dir": transform_artifacts_dir,
        "temporary_dir": temporary_dir,
        "gcs_location": f"gs://{bucket}/bq_tmp",
        "project": project,
    }

    logging.info(f"Transform pipeline args: {args}")
    etl.run_transform_pipeline(args)
    logging.info(f"Transform pipeline finished.")

    tft_output = tft.TFTransformOutput(transform_artifacts_dir)
    transform_feature_spec = tft_output.transformed_feature_spec()
    assert transform_feature_spec == EXPECTED_FEATURE_SPEC
