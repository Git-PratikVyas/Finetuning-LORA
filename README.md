
# Finetuning with LORA
- Finetuning pretrained decoder only model with [LORA](https://arxiv.org/abs/2106.09685) for (abstracrive ) summarization task.
- Push finetuned model to [Huggingface](https://huggingface.co/) hub for deployment on GKE ( or anyother cloud ). 
- Serve model using GPUs on GKE with [vLLM](https://docs.vllm.ai/en/latest/) for distributed inference.

# Table of Contents

1. [Introduction](#introduction)
2. [Prerequisite](#prerequisite)
3. [Deployment on GKE](#deployment)
4. [Serve the model](#serve-the-model)
5. [Appendix](#appendix)

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


## Appendix

Kubernetes Deployment YAML file in detail.

    Detailed Explanation

    1. metadata

    ```yaml
    metadata:
    name: vllm-gemma-deployment
    ```

    - __metadata__: Provides metadata for the deployment.
    - __name__: The name of the deployment. This name is used to identify the deployment within the Kubernetes cluster.

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

    - __spec__: Defines the desired state of the deployment, including the number of replicas, the pod template, and the container specifications.

    - __replicas__: Specifies the number of pod replicas to run. In this case, it is set to 1, meaning only one instance of the pod will be running.

    - __selector__: Defines the label selector to identify the pods managed by this deployment.
        - __matchLabels__: Specifies the labels that the pods must have to be managed by this deployment. In this case, the label is `app: mistral-summarizer-server`.

    - __template__: Defines the pod template used to create the pods.
        - __metadata__: Specifies metadata for the pods, including labels.
        - __labels__: Key-value pairs used to organize and select Kubernetes resources.
            - __app: mistral-summarizer-server__: Identifies the application to which the pod belongs.
            - __ai.gke.io/model: mistral-7B-instruct-v0.3__: Specifies the model being used by the pod.
            - __ai.gke.io/inference-server: vllm__: Indicates that the pod is using the vLLM inference server.
            - __examples.ai.gke.io/source: user-guide__: Provides additional metadata about the source of the configuration.

        - __spec__: Defines the specifications for the pod.
        - __containers__: Specifies the container configuration for the pod.
            - __name__: The name of the container (`inference-server`).
            - __image__: The Docker image to use for the container (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00`).
            - __resources__: Specifies the resource requests and limits for the container.
            - __requests__: The minimum amount of resources required for the container.
                - __cpu__: Requests 2 CPU cores.
                - __memory__: Requests 10 GiB of memory.
                - __ephemeral-storage__: Requests 10 GiB of ephemeral storage.
                - __nvidia.com/gpu__: Requests 1 NVIDIA GPU.
            - __limits__: The maximum amount of resources the container is allowed to use.
                - __cpu__: Limits the container to 2 CPU cores.
                - __memory__: Limits the container to 10 GiB of memory.
                - __ephemeral-storage__: Limits the container to 10 GiB of ephemeral storage.
                - __nvidia.com/gpu__: Limits the container to 1 NVIDIA GPU.

    ### Summary

    - __metadata__: Provides metadata for the deployment, including the name.
    - __name__: The name of the deployment.

    - __spec__: Defines the desired state of the deployment, including the number of replicas, the pod template, and the container specifications.
    - __replicas__: Specifies the number of pod replicas to run.
    - __selector__: Defines the label selector to identify the pods managed by this deployment.
        - __matchLabels__: Specifies the labels that the pods must have to be managed by this deployment.
    - __template__: Defines the pod template used to create the pods.
        - __metadata__: Specifies metadata for the pods, including labels.
        - __labels__: Key-value pairs used to organize and select Kubernetes resources.
            - __app: mistral-summarizer-server__: Identifies the application to which the pod belongs.
            - __ai.gke.io/model: mistral-7B-instruct-v0.3__: Specifies the model being used by the pod.
            - __ai.gke.io/inference-server: vllm__: Indicates that the pod is using the vLLM inference server.
            - __examples.ai.gke.io/source: user-guide__: Provides additional metadata about the source of the configuration.
        - __spec__: Defines the specifications for the pod.
        - __containers__: Specifies the container configuration for the pod.
            - __name__: The name of the container (`inference-server`).
            - __image__: The Docker image to use for the container (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00`).
            - __resources__: Specifies the resource requests and limits for the container.
            - __requests__: The minimum amount of resources required for the container.
                - __cpu__: Requests 2 CPU cores.
                - __memory__: Requests 10 GiB of memory.
                - __ephemeral-storage__: Requests 10 GiB of ephemeral storage.
                - __nvidia.com/gpu__: Requests 1 NVIDIA GPU.
            - __limits__: The maximum amount of resources the container is allowed to use.
                - __cpu__: Limits the container to 2 CPU cores.
                - __memory__: Limits the container to 10 GiB of memory.
                - __ephemeral-storage__: Limits the container to 10 GiB of ephemeral storage.
                - __nvidia.com/gpu__: Limits the container to 1 NVIDIA GPU.
