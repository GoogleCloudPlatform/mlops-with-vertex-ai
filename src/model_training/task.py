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
"""The entrypoint for the Vertex training job."""

import os
import sys
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse

from google.cloud import aiplatform as vertex_ai
import hypertune

from src.model_training import defaults, trainer, exporter


dirname = os.path.dirname(__file__)
dirname = dirname.replace("/model_training", "")
RAW_SCHEMA_LOCATION = os.path.join(dirname, "raw_schema/schema.pbtxt")
HYPERTUNE_METRIC_NAME = 'ACCURACY'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        type=str,
    )

    parser.add_argument(
        "--log-dir",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"),
        type=str,
    )

    parser.add_argument(
        "--train-data-dir",
        type=str,
    )

    parser.add_argument(
        "--eval-data-dir",
        type=str,
    )

    parser.add_argument(
        "--tft-output-dir",
        type=str,
    )

    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--batch-size", default=512, type=float)
    parser.add_argument("--hidden-units", default="64,32", type=str)
    parser.add_argument("--num-epochs", default=10, type=int)

    parser.add_argument("--project", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--staging-bucket", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    hyperparams = vars(args)
    hyperparams = defaults.update_hyperparams(hyperparams)
    logging.info(f"Hyperparameter: {hyperparams}")

    if args.experiment_name:
        vertex_ai.init(
            project=args.project,
            staging_bucket=args.staging_bucket,
            experiment=args.experiment_name,
        )

        logging.info(f"Using Vertex AI experiment: {args.experiment_name}")

        run_id = args.run_name
        if not run_id:
            run_id = f"run-gcp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        vertex_ai.start_run(run_id)
        logging.info(f"Run {run_id} started.")

        vertex_ai.log_params(hyperparams)

    classifier = trainer.train(
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        tft_output_dir=args.tft_output_dir,
        hyperparams=hyperparams,
        log_dir=args.log_dir,
    )

    val_loss, val_accuracy = trainer.evaluate(
        model=classifier,
        data_dir=args.eval_data_dir,
        raw_schema_location=RAW_SCHEMA_LOCATION,
        tft_output_dir=args.tft_output_dir,
        hyperparams=hyperparams,
    )
    
    
    # Report val_accuracy to Vertex hypertuner.
    logging.info(f'Reporting metric {HYPERTUNE_METRIC_NAME}={val_accuracy} to Vertex hypertuner...')
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=HYPERTUNE_METRIC_NAME,
        metric_value=val_accuracy,
        global_step=args.num_epochs * args.batch_size
    )

    # Log metrics in Vertex Experiments.
    logging.info(f'Logging metrics to Vertex Experiments...')
    if args.experiment_name:
        vertex_ai.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy})

    try:
        exporter.export_serving_model(
            classifier=classifier,
            serving_model_dir=args.model_dir,
            raw_schema_location=RAW_SCHEMA_LOCATION,
            tft_output_dir=args.tft_output_dir,
        )
    except:
        # Swallow Ignored Errors while exporting the model.
        pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {device_lib.list_local_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")
