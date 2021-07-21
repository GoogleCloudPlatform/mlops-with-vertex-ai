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
"""TFX pipeline configurations."""

import os

PROJECT = os.getenv("PROJECT", "ksalama-cloudml")
REGION = os.getenv("REGION", "us-central1")
GCS_LOCATION = os.getenv("GCS_LOCATION", "gs://ksalama-cloudml-us/ucaip_demo/")

ARTIFACT_STORE_URI = os.path.join(GCS_LOCATION, "tfx_artifacts")
MODEL_REGISTRY_URI = os.getenv(
    "MODEL_REGISTRY_URI",
    os.path.join(GCS_LOCATION, "model_registry"),
)

DATASET_DISPLAY_NAME = os.getenv("DATASET_DISPLAY_NAME", "chicago-taxi-tips")
MODEL_DISPLAY_NAME = os.getenv(
    "MODEL_DISPLAY_NAME", f"{DATASET_DISPLAY_NAME}-classifier"
)
PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{MODEL_DISPLAY_NAME}-train-pipeline")

ML_USE_COLUMN = "ml_use"
EXCLUDE_COLUMNS = ",".join(["trip_start_timestamp"])
TRAIN_LIMIT = os.getenv("TRAIN_LIMIT", "0")
TEST_LIMIT = os.getenv("TEST_LIMIT", "0")
SERVE_LIMIT = os.getenv("SERVE_LIMIT", "0")

NUM_TRAIN_SPLITS = os.getenv("NUM_TRAIN_SPLITS", "4")
NUM_EVAL_SPLITS = os.getenv("NUM_EVAL_SPLITS", "1")
ACCURACY_THRESHOLD = os.getenv("ACCURACY_THRESHOLD", "0.8")

USE_KFP_SA = os.getenv("USE_KFP_SA", "False")

TFX_IMAGE_URI = os.getenv(
    "TFX_IMAGE_URI", f"gcr.io/{PROJECT}/tfx-{DATASET_DISPLAY_NAME}:latest"
)

BEAM_RUNNER = os.getenv("BEAM_RUNNER", "DirectRunner")
BEAM_DIRECT_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
]
BEAM_DATAFLOW_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
    f"--region={REGION}",
    f"--runner={BEAM_RUNNER}",
]


TRAINING_RUNNER = os.getenv("TRAINING_RUNNER", "local")
AI_PLATFORM_TRAINING_ARGS = {
    "project": PROJECT,
    "region": REGION,
    "masterConfig": {"imageUri": TFX_IMAGE_URI},
}


SERVING_RUNTIME = os.getenv("SERVING_RUNTIME", "tf2-cpu.2-4")
SERVING_IMAGE_URI = f"gcr.io/cloud-aiplatform/prediction/{SERVING_RUNTIME}:latest"

BATCH_PREDICTION_BQ_DATASET_NAME = os.getenv(
    "BATCH_PREDICTION_BQ_DATASET_NAME", "playground_us"
)
BATCH_PREDICTION_BQ_TABLE_NAME = os.getenv(
    "BATCH_PREDICTION_BQ_TABLE_NAME", "chicago_taxitrips_prep"
)
BATCH_PREDICTION_BEAM_ARGS = {
    "runner": f"{BEAM_RUNNER}",
    "temporary_dir": os.path.join(GCS_LOCATION, "temp"),
    "gcs_location": os.path.join(GCS_LOCATION, "temp"),
    "project": PROJECT,
    "region": REGION,
    "setup_file": "./setup.py",
}
BATCH_PREDICTION_JOB_RESOURCES = {
    "machine_type": "n1-standard-2",
    #'accelerator_count': 1,
    #'accelerator_type': 'NVIDIA_TESLA_T4'
    "starting_replica_count": 1,
    "max_replica_count": 10,
}
DATASTORE_PREDICTION_KIND = f"{MODEL_DISPLAY_NAME}-predictions"

ENABLE_CACHE = os.getenv("ENABLE_CACHE", "0")
UPLOAD_MODEL = os.getenv("UPLOAD_MODEL", "1")

os.environ["PROJECT"] = PROJECT
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["TFX_IMAGE_URI"] = TFX_IMAGE_URI
os.environ["MODEL_REGISTRY_URI"] = MODEL_REGISTRY_URI
