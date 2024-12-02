
# Finetuning with LORA
- Finetuning pretrained decoder only model with LORA for (abstracrive ) summarization task.
- Push finetuned model to Huggingface hub for deployment on GKE ( or anyother cloud ).
- Use [vLLM](https://docs.vllm.ai/en/latest/) inference server on GKE to serve model for distributed inference.  

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
- Machine with GPU (T4 or higher) 
- Huggingface account and token to load pre-trained model and push finetuned model to hub.
- Weights & Biases (W&B) account and token to integrate and publish all training/eval matrices for monitoring and evaluation.


## Deployment
Instructions for installation.

## Usage
How to use the project.

## Contributing
Guidelines for contributing to the project.