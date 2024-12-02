
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
```shell
        gcloud config set project PROJECT_ID
        export PROJECT_ID=$(gcloud config get project)
        export REGION=REGION
        export CLUSTER_NAME=vllm
        export HF_TOKEN=HF_TOKEN
 ```       
- Create a GKE cluster and node pool
```shell
        gcloud container clusters create-auto ${CLUSTER_NAME} \
        --project=${PROJECT_ID} \
        --region=${REGION} \
        --release-channel=rapid
 ```  
- Create a Kubernetes secret for Hugging Face credentials
```shell
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
```
```shell
    kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=$HF_TOKEN \
    --dry-run=client -o yaml | kubectl apply -f -
```

- Deploy a vLLM to your cluster.
deploy the vLLM container to serve ```Prat/Mistral-7B-Instruct-v0.3_summarizer_v1```

1. Create the following vllm-2-2b-it.yaml manifest:
```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: vllm-gemma-deployment
    spec:
    replicas: 1
    selector:
        matchLabels:
        app: mistral-summarizer-server
    template:
        metadata:
        labels:
            app: mistral-summarizer-server
            ai.gke.io/model: mistral-7B-instruct-v0.3
            ai.gke.io/inference-server: vllm
            examples.ai.gke.io/source: user-guide
        spec:
        containers:
        - name: inference-server
            image: us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00
            resources:
            requests:
                cpu: "2"
                memory: "10Gi"
                ephemeral-storage: "10Gi"
                nvidia.com/gpu: "1"
            limits:
                cpu: "2"
                memory: "10Gi"
                ephemeral-storage: "10Gi"
                nvidia.com/gpu: "1"
            command: ["python3", "-m", "vllm.entrypoints.api_server"]
            args:
            - --model=$(MODEL_ID)
            - --tensor-parallel-size=1
            env:
            - name: MODEL_ID
            value: Prat/Mistral-7B-Instruct-v0.3_summarizer_v1
            - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
                secretKeyRef:
                name: hf-secret
                key: hf_api_token
            volumeMounts:
            - mountPath: /dev/shm
            name: dshm
        volumes:
        - name: dshm
            emptyDir:
                medium: Memory
        nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-l4
            cloud.google.com/gke-gpu-driver-version: latest
    ---
    apiVersion: v1
    kind: Service
    metadata:
    name: llm-service
    spec:
    selector:
        app: mistral-summarizer-server
    type: ClusterIP
    ports:
        - protocol: TCP
        port: 8000
        targetPort: 8000
```

2. Apply the manifest:

once you apply this command, A Pod in the cluster downloads the model weights from Hugging Face and starts the serving engine.
```shell
    kubectl apply -f vllm-2-2b-it.yaml
```

3. Wait for the Deployment to be available:
```shell 
    kubectl wait --for=condition=Available --timeout=700s deployment/vllm-gemma-deployment
```


- Use vLLM to serve the model through curl and a web chat interface.


## Usage
How to use the project.

## Contributing
Guidelines for contributing to the project.