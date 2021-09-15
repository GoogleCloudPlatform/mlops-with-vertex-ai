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
"""Utilities for generating BigQuery data querying scirpts."""


from google.cloud import aiplatform as vertex_ai


def _get_source_query(bq_dataset_name, bq_table_name, ml_use, limit=None):
    query = f"""
    SELECT 
        IF(trip_month IS NULL, -1, trip_month) trip_month,
        IF(trip_day IS NULL, -1, trip_day) trip_day,
        IF(trip_day_of_week IS NULL, -1, trip_day_of_week) trip_day_of_week,
        IF(trip_hour IS NULL, -1, trip_hour) trip_hour,
        IF(trip_seconds IS NULL, -1, trip_seconds) trip_seconds,
        IF(trip_miles IS NULL, -1, trip_miles) trip_miles,
        IF(payment_type IS NULL, 'NA', payment_type) payment_type,
        IF(pickup_grid IS NULL, 'NA', pickup_grid) pickup_grid,
        IF(dropoff_grid IS NULL, 'NA', dropoff_grid) dropoff_grid,
        IF(euclidean IS NULL, -1, euclidean) euclidean,
        IF(loc_cross IS NULL, 'NA', loc_cross) loc_cross"""
    if ml_use:
        query += f""",
        tip_bin
    FROM {bq_dataset_name}.{bq_table_name} 
    WHERE ML_use = '{ml_use}'
    """
    else:
        query += f"""
    FROM {bq_dataset_name}.{bq_table_name} 
    """
    if limit:
        query += f"LIMIT {limit}"

    return query


def get_training_source_query(
    project, region, dataset_display_name, ml_use, limit=None
):
    vertex_ai.init(project=project, location=region)
    
    dataset = vertex_ai.TabularDataset.list(
        filter=f"display_name={dataset_display_name}", order_by="update_time"
    )[-1]
    bq_source_uri = dataset.gca_resource.metadata["inputConfig"]["bigquerySource"][
        "uri"
    ]
    _, bq_dataset_name, bq_table_name = bq_source_uri.replace("g://", "").split(".")

    return _get_source_query(bq_dataset_name, bq_table_name, ml_use, limit)


def get_serving_source_query(bq_dataset_name, bq_table_name, limit=None):

    return _get_source_query(bq_dataset_name, bq_table_name, ml_use=None, limit=limit)
