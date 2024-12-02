
# Finetuning with LORA and Serve model on GKE using vLLM
- Finetuning pretrained decoder only model with [LORA](https://arxiv.org/abs/2106.09685) for (abstracrive ) summarization task.
- Push finetuned model to [Huggingface](https://huggingface.co/) hub for deployment on GKE ( or anyother cloud ). 
- Serve model using GPUs on GKE with [vLLM](https://docs.vllm.ai/en/latest/) for distributed inference.

# Table of Contents

1. [Introduction](#introduction)
2. [Prerequisite](#prerequisite)
3. [Deployment on GKE](#deployment)
4. [Serve the model](#serve-the-model)
5. [Autoscaling for better Latency and Throughput on GKE](#autoscaling-for-better-latency-and-throughput-on-gke)
6. [GPU utilization on GKE](gpu-utilization-on-gke)
7. [Appendix-Kubernetes Deployment Explanation](#appendix-kubernetes-deployment-explanation)

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

1. Create the following vllm-3-7b-it.yaml manifest:
```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: vllm-mistral-deployment
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
    kubectl apply -f vllm-3-7b-it.yaml
```

3. Wait for the Deployment to be available:
```shell 
    kubectl wait --for=condition=Available --timeout=700s deployment/vllm-mistral-deployment
```

## Serve the model
1. Set up port forwarding
```shell
    kubectl port-forward service/llm-service 8000:8000
```
2. Interact with the model using curl
```shell
        USER_PROMPT="
        user:\nSummarise dialogue in one sentence.\n
        dialogue: Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)
        summary:"

        curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d @- <<EOF
        {
            "prompt": "<start_of_turn>user\n${USER_PROMPT}<end_of_turn>\n",
            "temperature": 0.90,
            "top_p": 1.0,
            "max_tokens": 128
        }
        EOF
```
2. you can also create UI to interact with the model.


## Autoscaling for better Latency and Throughput on GKE
Use [Horizontal Pod Scaling (HPS)](https://cloud.google.com/kubernetes-engine/docs/concepts/horizontalpodautoscaler) to improve latency and throughput.
Important matrices for HPS are
1. Queue Size: First option to choose if latency target can be met with queue size autoscaling.
2. Batch Size: Good for latency sensitive workload and requirement are not met with queue size autoscaling.
3. GPU Memory Usage: Good indicator to upscale resources.


## GPU utilization on GKE
There are three technique through which GPU can be utilised optimaly.
1. Time-sharing GPU
2. Multi-instance GPU
3. NVIDIA MPS


## Appendix Kubernetes Deployment Explanation

Kubernetes Deployment YAML file in detail.

Detailed Explanation

    1. metadata

```yaml
    metadata:
    name: vllm-mistral-deployment
```

    - metadata: Provides metadata for the deployment.
    - name: The name of the deployment. This name is used to identify the deployment within the Kubernetes cluster.

    2. spec

```yaml
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
```

    - spec: Defines the desired state of the deployment, including the number of replicas, the pod template, and the container specifications.

    - replicas: Specifies the number of pod replicas to run. In this case, it is set to 1, meaning only one instance of the pod will be running.

    - selector: Defines the label selector to identify the pods managed by this deployment.
        - matchLabels: Specifies the labels that the pods must have to be managed by this deployment. In this case, the label is `app: mistral-summarizer-server`.

    - template: Defines the pod template used to create the pods.
        - metadata: Specifies metadata for the pods, including labels.
        - labels: Key-value pairs used to organize and select Kubernetes resources.
            - app: mistral-summarizer-server: Identifies the application to which the pod belongs.
            - ai.gke.io/model: mistral-7B-instruct-v0.3: Specifies the model being used by the pod.
            - ai.gke.io/inference-server: vllm: Indicates that the pod is using the vLLM inference server.
            - examples.ai.gke.io/source: user-guide: Provides additional metadata about the source of the configuration.

        - spec: Defines the specifications for the pod.
        - containers: Specifies the container configuration for the pod.
            - name: The name of the container (`inference-server`).
            - image: The Docker image to use for the container (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00`).
            - resources: Specifies the resource requests and limits for the container.
            - requests: The minimum amount of resources required for the container.
                - cpu: Requests 2 CPU cores.
                - memory: Requests 10 GiB of memory.
                - ephemeral-storage: Requests 10 GiB of ephemeral storage.
                - nvidia.com/gpu: Requests 1 NVIDIA GPU.
            - limits: The maximum amount of resources the container is allowed to use.
                - cpu: Limits the container to 2 CPU cores.
                - memory: Limits the container to 10 GiB of memory.
                - ephemeral-storage: Limits the container to 10 GiB of ephemeral storage.
                - nvidia.com/gpu: Limits the container to 1 NVIDIA GPU.

    - Summary

    - metadata: Provides metadata for the deployment, including the name.
    - name: The name of the deployment.
    - spec: Defines the desired state of the deployment, including the number of replicas, the pod template, and the container specifications.
    - replicas: Specifies the number of pod replicas to run.
    - selector: Defines the label selector to identify the pods managed by this deployment.
        - matchLabels: Specifies the labels that the pods must have to be managed by this deployment.
    - template: Defines the pod template used to create the pods.
        - metadata: Specifies metadata for the pods, including labels.
        - labels: Key-value pairs used to organize and select Kubernetes resources.
            - app: mistral-summarizer-server: Identifies the application to which the pod belongs.
            - ai.gke.io/model: mistral-7B-instruct-v0.3: Specifies the model being used by the pod.
            - ai.gke.io/inference-server: vllm: Indicates that the pod is using the vLLM inference server.
            - examples.ai.gke.io/source: user-guide: Provides additional metadata about the source of the configuration.
        - spec: Defines the specifications for the pod.
        - containers: Specifies the container configuration for the pod.
            - name: The name of the container (`inference-server`).
            - image: The Docker image to use for the container (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00`).
            - resources: Specifies the resource requests and limits for the container.
            - requests: The minimum amount of resources required for the container.
                - cpu: Requests 2 CPU cores.
                - memory: Requests 10 GiB of memory.
                - ephemeral-storage: Requests 10 GiB of ephemeral storage.
                - nvidia.com/gpu: Requests 1 NVIDIA GPU.
            - limits: The maximum amount of resources the container is allowed to use.
                - cpu: Limits the container to 2 CPU cores.
                - memory: Limits the container to 10 GiB of memory.
                - ephemeral-storage: Limits the container to 10 GiB of ephemeral storage.
                - nvidia.com/gpu: Limits the container to 1 NVIDIA GPU.
