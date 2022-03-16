# MLOps with Vertex AI

This example implements the end-to-end [MLOps process](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) using [Vertex AI](https://cloud.google.com/vertex-ai) platform and [Smart Analytics](https://cloud.google.com/solutions/smart-analytics) technology capabilities. The example uses [Keras](https://keras.io/) to implement the ML model, [TFX](https://www.tensorflow.org/tfx) to implement the training pipeline, and [Model Builder SDK](https://github.com/googleapis/python-aiplatform/tree/569d4cd03e888fde0171f7b0060695a14f99b072/google/cloud/aiplatform) to interact with Vertex AI.

<p align="center">
    <img src="mlops.png" alt="MLOps lifecycle" width="400"/>
</p>


## Getting started

1. [Setup your MLOps environment](provision) on Google Cloud.
2. Start your AI Notebook instance.
3. Open the JupyterLab then open a new Terminal
4. Clone the repository to your AI Notebook instance:
    ```
    git clone https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai.git
    cd mlops-with-vertex-ai
    ```
5. Install the required Python packages:
    ```
    pip install tfx==1.2.0 --user
    pip install -r requirements.txt
    ```
    ---
    **NOTE**: You can ignore the pip dependencies issues. These will be fixed when upgrading to subsequent TFX version.
    
    ---
6. Upgrade the `gcloud` components:
    ```
   sudo apt-get install google-cloud-sdk
   gcloud components update
   ```

## Dataset Management

The [Chicago Taxi Trips](https://pantheon.corp.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips) dataset is one of [public datasets hosted with BigQuery](https://cloud.google.com/bigquery/public-data/), which includes taxi trips from 2013 to the present, reported to the City of Chicago in its role as a regulatory agency. The task is to predict whether a given trip will result in a tip > 20%.

The [01-dataset-management](01-dataset-management.ipynb) notebook covers:

1. Performing exploratory data analysis on the data in `BigQuery`.
2. Creating `Vertex AI` Dataset resource using the Python SDK.
3. Generating the schema for the raw data using [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv).


## ML Development

We experiment with creating a [Custom Model](https://cloud.google.com/ai-platform-unified/docs/training/create-model-custom-training) using [02-experimentation](02-experimentation.ipynb) notebook, which covers:

1. Preparing the data using `Dataflow`.
2. Implementing a `Keras` classification model.
3. Training the `Keras` model with `Vertex AI` using a [pre-built container](https://cloud.google.com/ai-platform-unified/docs/training/pre-built-containers).
4. Upload the exported model from `Cloud Storage` to `Vertex AI`.
5. Extract and visualize experiment parameters from [Vertex AI Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction).
6. Use `Vertex AI` for [hyperparameter tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview).

We use [Vertex TensorBoard](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview) 
and [Vertex ML Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction) to  track, visualize, and compare ML experiments.

In addition, the training steps are formalized by implementing a [TFX pipeline](https://www.tensorflow.org/tfx).
The [03-training-formalization](03-training-formalization.ipynb) notebook covers implementing and testing the pipeline components interactively.

## Training Operationalization

The [04-pipeline-deployment](04-pipeline-deployment.ipynb) notebook covers executing the CI/CD steps for the training pipeline deployment using [Cloud Build](https://cloud.google.com/build/docs/overview). The CI/CD routine is defined in the [pipeline-deployment.yaml](build/pipeline-deployment.yaml) file, and consists of the following steps:

1. Clone the repository to the build environment.
2. Run unit tests.
3. Run a local e2e test of the `TFX` pipeline.
4. Build the ML container image for pipeline steps.
5. Compile the pipeline.
6. Upload the pipeline to `Cloud Storage`.

## Continuous Training

After testing, compiling, and uploading the pipeline definition to `Cloud Storage`, the pipeline is executed with respect to a trigger. 
We use [Cloud Functions](https://cloud.google.com/functions) and [Cloud Pub/Sub](https://cloud.google.com/pubsub) as a triggering mechanism.
The `Cloud Function` listens to the `Pub/Sub` topic, and runs the training pipeline given a message sent to the `Pub/Sub` topic.
The `Cloud Function` is implemented in [src/pipeline_triggering](src/pipeline_triggering). 

The [05-continuous-training](05-continuous-training.ipynb) notebook covers:

1. Creating a Cloud `Pub/Sub` topic.
2. Deploying a `Cloud Function`.
3. Triggering the pipeline.

The end-to-end TFX training pipeline implementation is in the [src/pipelines](src/tfx_pipelines) directory, which covers the following steps:

1. Receive hyper-parameters using `hyperparam_gen` custom python component.
2. Extract data from `BigQuery` using `BigQueryExampleGen` component.
3. Validate the raw data using `StatisticsGen` and `ExampleValidator` component.
4. Process the data using on `Dataflow` `Transform` component.
5. Train a custom model with `Vertex AI` using `Trainer` component.
6. Evaluate and validate the custom model using `ModelEvaluator` component.
7. Save the blessed to model registry location in `Cloud Storage` using `Pusher` component.
8. Upload the model to `Vertex AI` using `vertex_model_pusher` custom python component.


## Model Deployment

The [06-model-deployment](06-model-deployment.ipynb) notebook covers executing the CI/CD steps for the model deployment using [Cloud Build](https://cloud.google.com/build/docs/overview). The CI/CD routine is defined in [build/model-deployment.yaml](build/model-deployment.yaml)
file, and consists of the following steps:

2. Test model interface.
3. Create an endpoint in `Vertex AI`.
4. Deploy the model to the `endpoint`.
5. Test the `Vertex AI` endpoint.

## Prediction Serving

We serve the deployed model for prediction. 
The [07-prediction-serving](07-prediction-serving.ipynb) notebook covers:

1. Use the `Vertex AI` endpoint for online prediction.
2. Use the `Vertex AI` uploaded model for batch prediction.
3. Run the batch prediction using `Vertex Pipelines`.

## Model Monitoring

After a model is deployed in for prediction serving, continuous monitoring is set up to ensure that the model continue to perform as expected.
The [08-model-monitoring](08-model-monitoring.ipynb) notebook covers configuring [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview?hl=nn) for skew and drift detection:

1. Set skew and drift threshold.
2. Create a monitoring job for all the models under and endpoint.
3. List the monitoring jobs.
4. List artifacts produced by monitoring job.
5. Pause and delete the monitoring job.


## Metadata Tracking

You can view the parameters and metrics logged by your experiments, as well as the artifacts and metadata stored by 
your `Vertex Pipelines` in [Cloud Console](https://console.cloud.google.com/vertex-ai/metadata).

## Disclaimer

This is not an official Google product but sample code provided for an educational purpose.

---

Copyright 2021 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.






