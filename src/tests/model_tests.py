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
"""Test model functions."""

import sys
import logging
from src.model_training import model, defaults

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

EXPECTED_HYPERPARAMS_KEYS = [
    "hidden_units",
    "learning_rate",
    "batch_size",
    "num_epochs",
]


def test_hyperparams_defaults():
    hyperparams = {"hidden_units": [64, 32]}

    hyperparams = defaults.update_hyperparams(hyperparams)
    assert set(hyperparams.keys()) == set(EXPECTED_HYPERPARAMS_KEYS)


def test_create_binary_classifier():

    hyperparams = hyperparams = defaults.update_hyperparams(dict())