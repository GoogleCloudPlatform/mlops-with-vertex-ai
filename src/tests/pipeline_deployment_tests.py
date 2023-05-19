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
"""Test training pipeline using local runner."""

import sys
import os
import logging

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

MLMD_SQLLITE = "mlmd.sqllite"
NUM_EPOCHS = 1
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN_UNITS = "128,128"


def test_e2e_pipeline():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    dataset_display_name = os.getenv("DATASET_DISPLAY_NAME")
    gcs_location = os.getenv("GCS_LOCATION")
    model_registry = os.getenv("MODEL_REGISTRY_URI")
    upload_model = os.getenv("UPLOAD_MODEL")
