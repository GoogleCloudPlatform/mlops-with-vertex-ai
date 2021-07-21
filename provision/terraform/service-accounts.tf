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

# Create Vertex Training service account
resource "google_service_account" "training_sa" {
    project       = module.project-services.project_id
    account_id   = var.training_sa_name
    display_name = "Vertex Training service account"
}

# Create Vertex Training SA role bindings
resource "google_project_iam_member" "training_sa_role_bindings" {
    project       = module.project-services.project_id
    for_each = toset(var.training_sa_roles)
    member   = "serviceAccount:${google_service_account.training_sa.email}"
    role     = "roles/${each.value}"
}

# Create Vertex Pipelines service account
resource "google_service_account" "pipelines_sa" {
    project       = module.project-services.project_id
    account_id   = var.pipelines_sa_name
    display_name = "Vertex Pipelines account name"
}

# Create Vertex Pipelines SA role bindings
resource "google_project_iam_member" "role_bindings" {
    project       = module.project-services.project_id
    for_each = toset(var.pipelines_sa_roles)
    member   = "serviceAccount:${google_service_account.pipelines_sa.email}"
    role     = "roles/${each.value}"
}
