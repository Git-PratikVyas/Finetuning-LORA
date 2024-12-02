
# Finetuning with LORA
- Finetuning pretrained decoder only model with [LORA](https://arxiv.org/abs/2106.09685) for (abstracrive ) summarization task.
- Push finetuned model to [Huggingface](https://huggingface.co/) hub for deployment on GKE ( or anyother cloud ). 
- Serve model using GPUs on GKE with [vLLM](https://docs.vllm.ai/en/latest/) for distributed inference.

# Table of Contents

1. [Introduction](#introduction)
2. [Prerequisite](#prerequisite)
2. [Deployment on GKE](#deployment)
3. [Usage](#usage)
4. [Contributing](#contributing)

## Introduction
Finetuned two pretrained models 
[Mistral-7B-Instruct-v0.3]( https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 ) and [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct ) on [Samsum]( https://paperswithcode.com/paper/samsum-corpus-a-human-annotated-dialogue-1 ) database.
Integrate and publish all training/eval matrices to [Weights & Biases (W&B)]( https://wandb.ai/home ) for tracking, monitoring, and collaboration.

Evaluate finetuned model on Rouge score and publish better model ( Mistral-7B-Instruct-v0.3 ) to [Huggingface hub]( https://huggingface.co/Prat/Mistral-7B-Instruct-v0.3_summarizer_v1 ) for deployment on GKE.


## Prerequisite
- Machine with ***GPU*** (T4 or higher).
- ***Google cloud GKE*** to server model. 
- ***Huggingface account*** and token to load pre-trained model and push finetuned model to hub.
- ***Weights & Biases (W&B)*** account and token to integrate and publish all training/eval matrices for monitoring and evaluation.



## Deployment
- Prepare your environment with a GKE cluster in Autopilot mode.
- Make sure [GPU quota](https://cloud.google.com/compute/resource-usage#gpu_quota) are available to your project.
- Set the default environment variables
```command
        gcloud config set project PROJECT_ID
        export PROJECT_ID=$(gcloud config get project)
        export REGION=REGION
        export CLUSTER_NAME=vllm
        export HF_TOKEN=HF_TOKEN
- Create a GKE cluster and node pool
        gcloud container clusters create-auto ${CLUSTER_NAME} \
        --project=${PROJECT_ID} \
        --region=${REGION} \
        --release-channel=rapid


- Deploy a vLLM container to your cluster.
- Use vLLM to serve the model through curl and a web chat interface.


## Usage
How to use the project.

## Contributing
Guidelines for contributing to the project.