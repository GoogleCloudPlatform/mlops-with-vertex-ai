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

locals {
    image_project = "deeplearning-platform-release"
}

data "google_compute_network" "vm_network" {
    project = module.project-services.project_id
    name    = var.network_name

    depends_on = [
        module.project-services
    ]
}

data "google_compute_subnetwork" "vm_subnetwork" {
    project = module.project-services.project_id
    name   = var.subnet_name
    region = var.subnet_region

    depends_on = [
        module.project-services
    ]
}

resource "google_notebooks_instance" "notebook_instance" {
    project          = module.project-services.project_id
    name             = "${var.name_prefix}-notebook"
    machine_type     = var.machine_type
    location         = var.zone

    network = data.google_compute_network.vm_network.id
    subnet  = data.google_compute_subnetwork.vm_subnetwork.id

    vm_image {
        project      = local.image_project
        image_family = var.image_family
    }

    dynamic accelerator_config {
      for_each = var.gpu_type != null ? [1] : []
      content {
          type = var.gpu_type
          core_count = var.gpu_count
      }
    }

    install_gpu_driver  = var.install_gpu_driver

    boot_disk_size_gb   = var.boot_disk_size
}
