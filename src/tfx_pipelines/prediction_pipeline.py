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
"""TFX prediction pipeline definition."""

import os
import sys
import json
import logging

from tfx.orchestration import pipeline, data_types
from ml_metadata.proto import metadata_store_pb2

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.tfx_pipelines import config
from src.tfx_pipelines import components as custom_components
from src.common import datasource_utils


def create_pipeline(
    pipeline_root: str,
    metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,
):

    # Get source query.
    sql_query = datasource_utils.get_serving_source_query(
        bq_dataset_name=config.BATCH_PREDICTION_BQ_DATASET_NAME,
        bq_table_name=config.BATCH_PREDICTION_BQ_TABLE_NAME,
        limit=int(config.SERVE_LIMIT),
    )

    bigquery_data_gen = custom_components.bigquery_data_gen(
        sql_query=sql_query,
        output_data_format="jsonl",
        beam_args=json.dumps(config.BATCH_PREDICTION_BEAM_ARGS),
    )

    vertex_batch_prediction = custom_components.vertex_batch_prediction(
        project=config.PROJECT,
        region=config.REGION,
        model_display_name=config.MODEL_DISPLAY_NAME,
        instances_format="jsonl",
        predictions_format="jsonl",
        job_resources=json.dumps(config.BATCH_PREDICTION_JOB_RESOURCES),
        serving_dataset=bigquery_data_gen.outputs["serving_dataset"],
    )

    datastore_prediction_writer = custom_components.datastore_prediction_writer(
        datastore_kind=config.DATASTORE_PREDICTION_KIND,
        predictions_format="jsonl",
        beam_args=json.dumps(config.BATCH_PREDICTION_BEAM_ARGS),
        prediction_results=vertex_batch_prediction.outputs["prediction_results"],
    )

    pipeline_components = [
        bigquery_data_gen,
        vertex_batch_prediction,
        datastore_prediction_writer,
    ]

    logging.info(
        f"Pipeline components: {[component.id for component in pipeline_components]}"
    )

    beam_pipeline_args = config.BEAM_DIRECT_PIPELINE_ARGS
    if config.BEAM_RUNNER == "DataflowRunner":
        beam_pipeline_args = config.BEAM_DATAFLOW_PIPELINE_ARGS

    logging.info(f"Beam pipeline args: {beam_pipeline_args}")

    return pipeline.Pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config,
        enable_cache=int(config.ENABLE_CACHE),
    )
