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

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

OUTPUT_DIR = "test_etl_output_dir"
ML_USE = "UNASSIGNED"
LIMIT = 100


def test_transform_pipeline():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    bucket = os.getenv("BUCKET")
    dataset_display_name = os.getenv("DATASET_DISPLAY_NAME")
