
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


variable "project_id" {
    description = "The GCP project ID"
    type        = string
}

variable "region" {
    description = "The region for the GCS bucket and Artifact Registry"
    type        = string
    default     = null
}

variable "zone" {
    description = "The zone for a Vertex Notebook instance"
    type        = string
}

variable "name_prefix" {
    description = "The name prefix to add to the resource names"
    type        = string
}

variable "machine_type" {
    description = "The Notebook instance's machine type"
    type        = string
}

variable "network_name" {
  description = "The network name for the Notebook instance"
  type        = string
  default     = "default"
}

variable "subnet_name" {
  description = "The subnet name for the Notebook instance"
  type        = string
  default     = "default"
}

variable "subnet_region" {
    description = "The region for the Notebook subnet"
    type        = string
    default     = "us-central1"
}

variable "boot_disk_size" {
    description = "The size of the boot disk"
    default     = 200
}

variable "image_family" {
    description = "A Deep Learning image family for the Notebook instance"
    type        = string
    default     = "tf-2-4-cpu"
}

variable "gpu_type" {
    description = "A GPU type for the Notebook instance"
    type        = string
    default     = null
}

variable "gpu_count" {
    description = "A GPU count for the Notebook instance"
    type        = string
    default     = null
}

variable "install_gpu_driver" {
    description = "Whether to install GPU driver"
    type        = bool
    default     = false
}

variable "force_destroy" {
    description = "Whether to remove the bucket on destroy"
    type        = bool
    default     = false
}

variable "training_sa_roles" {
  description = "The roles to assign to the Vertex Training service account"
  default = [
    "storage.admin",
    "aiplatform.user",
    "bigquery.admin"
    ] 
}

variable "pipelines_sa_roles" {
  description = "The roles to assign to the Vertex Pipelines service account"
  default = [    
    "storage.admin", 
    "bigquery.admin", 
    "aiplatform.user"
  ]
}

variable "training_sa_name" {
    description = "Vertex training service account name."
    default = "training-sa"
}

variable "pipelines_sa_name" {
    description = "Vertex pipelines service account name."
    default = "pipelines-sa"
}
