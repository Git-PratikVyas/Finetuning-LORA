# List of Models finetuned for cancer classification
1. [meta-llama/Llama-3.1-8B-Instruct (CausalLM)](https://github.com/Git-PratikVyas/Finetuning-LORA/blob/main/CancerClassification/Llama-3.1-8B-Instruct_cancer_classification_finetuning.ipynb)
2. [distilbert (SequenceClassification)](https://github.com/Git-PratikVyas/Finetuning-LORA/blob/main/CancerClassification/distilbert_cancer_classification_finetuning.ipynb)


# other details
- Finetuned with LORA, serving on GKE using vLLM
- Used [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index) for distributed training
- Pushed finetuned model to [Huggingface Hub](https://huggingface.co/Prat/Llama-3.1-8B-Instruct-ft-summarizer-v1) hub for deployment on GKE ( or anyother cloud ). 
- Serve model using GPUs on GKE with [vLLM](https://docs.vllm.ai/en/latest/) for distributed inference.
- Find my blog on [LLM inference](https://medium.com/@pratik.vyas_10544/llm-inference-vllm-tgi-tensorrt-17872f7df1f5)
- Find my blog on [LLM Transformer(encoder,decoder)](https://medium.com/@pratik.vyas_10544/transformer-encoder-part-1-1d926707f74b)

![screenshot](../other/Picture1.png)
![screenshot](../other/Picture2.png)
![screenshot](../other/Picture3.png)
![screenshot](../other/Picture4.png)

# Model Deployment on GKE

1. [Introduction](#introduction)
2. [Prerequisite](#prerequisite)
3. [Deployment on GKE](#deployment)
4. [Serve the model](#serve-the-model)
5. [GKE ```HPA``` for better latency and throughput](#GKE-HPA-for-better-latency-and-throughput)
6. [Better latency and throughput using vLLM](#Better-latency-and-throughput-using-vLLM)
7. [GPU utilization on GKE](#gpu-utilization-on-gke)
8. [Appendix-Kubernetes Deployment Explanation](#appendix-kubernetes-deployment-explanation)

## Introduction
Distributed finetuned pretrained model meta-llama/Llama-3.1-8B-Instruct on cancer dataset using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index).


Evaluate finetuned model using confusion matrix ( Accuract,F1-score ) and push model ( Llama-3.1-8B-Inst_cancer_classification_finetuned ) to [Huggingface hub](https://huggingface.co/Prat/Llama-3.1-8B-Instruct-ft-summarizer-v1) for deployment on GKE.


## Prerequisite
- Machine with ***GPU*** (T4 or higher).
- ***Google cloud GKE*** to serve model. 
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
deploy the vLLM container to serve ```Prat/Llama-3.1-8B-Instruct_ft_summarizer_v1```

1. Create the following vllm-3-7b-it.yaml manifest:
```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: vllm-cancer-classification-deployment
    spec:
    replicas: 1
    selector:
        matchLabels:
        app: cancer-classification-server
    template:
        metadata:
        labels:
            app: cancer-classification-server
            ai.gke.io/model: Llama-3.1-8B-Instruct
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
            value: Prat/Llama-3.1-8B-Instruct_ft_summarizer_v1
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
        app: cancer-classification-server
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
    kubectl wait --for=condition=Available --timeout=700s deployment/vllm-cancer-classification-deployment
```

## Serve the model
1. Set up port forwarding
```shell
    kubectl port-forward service/llm-service 8000:8000
```
2. Interact with the model using curl
```shell
        USER_PROMPT="
        Given the following text, classify if it is related to cancer or not. If text is related to cancer, Label as 'True' otherwise Label as 'False'.
        
        Text:
        Title: Similarities and differences of interstitial lung disease associated with pathogenic variants in SFTPC and ABCA3 in adults.
        Abstract: Variants in surfactant genes SFTPC or ABCA3 are responsible for interstitial lung disease (ILD) in children and adults, with few studies in adults.
        We conducted a multicentre retrospective study of all consecutive adult patients diagnosed with ILD associated with variants in SFTPC or ABCA3 in the French rare pulmonary diseases network, OrphaLung. Variants and chest computed tomography (CT) features were centrally reviewed.

        Label: "

        curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d @- <<EOF
        {
            "prompt": "<start_of_turn>user\n${USER_PROMPT}<end_of_turn>\n",
            "temperature": 0.10,
            "max_tokens": 3
        }
        EOF
```
2. you can also create UI to interact with the model.


## GKE HPA for better latency and throughput
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

## Better latency and throughput using vLLM

  **1. <ins>vLLM Speculative Decoding:<ins>**

  vLLM Speculative Decoding for better latency and throughput during inference
  Speculative decoding addresses the inherent latency in traditional autoregressive decoding methods, where each token is generated sequentially based on all previous tokens. Instead, it allows for the simultaneous prediction of multiple tokens, thereby accelerating the inference process.

  - Mechanism:
    - In speculative decoding, a **draft model** (often smaller and faster) generates several potential future tokens in parallel during each decoding step. These tokens are then verified by the larger, more complex target model.
    - This two-step process consists of:
      - **Drafting**: The draft model quickly proposes multiple tokens.
      - **Verification**: The target model evaluates these proposals and selects the valid ones based on its criteria.

  - Efficiency Gains:
    - By generating multiple tokens at once, speculative decoding can significantly reduce the time taken for each inference step. This is particularly beneficial in memory-bound scenarios where traditional approaches may struggle due to high memory read/write latencies.

  Benefits of Speculative Decoding in vLLM

  - Increased Throughput:
    Speculative decoding can improve throughput by allowing the model to process more tokens per forward pass compared to standard methods that generate one token at a time. This can lead to speedups of **3-6 times** depending on the implementation and model configuration.

  - Reduced Latency:
    The technique minimizes inter-token latency by allowing multiple tokens to be processed simultaneously, which is crucial for applications requiring real-time responses.

  - Adaptability:
    Speculative decoding can be adapted to various configurations, such as using different draft models or adjusting the number of speculative tokens generated based on system load and requirements.

  - Improved Resource Utilization:
    By optimizing how models utilize GPU resources, speculative decoding enhances the overall efficiency of LLM inference, making it more feasible to deploy large models in production environments with limited computational resources.

  - Change deployment as below

```yaml
args:
            - --model=$(MODEL_ID)
            - --tensor-parallel-size=1
            - --num-speculative-tokens=5  # Specify the number of speculative tokens to generate
            - --speculative-model=facebook/opt-125m  # Draft model for speculation
```

  **2. <ins>Multiple LoRA (Low-Rank Adaptation) adapters with vLLM:</ins>**

  Allows for efficient specialization of large language models (LLMs) for various tasks without the need for unloading and reloading adapters, which can degrade user experience. Here’s a comprehensive guide on how to implement multi-LoRA functionality in vLLM based on the search results.
  - Multi-LoRA enables the simultaneous use of different LoRA adapters, allowing a single model to handle various tasks (e.g., translation, classification) without noticeable delays between requests.
  - This approach optimizes resource utilization and improves response times, making it suitable for applications needing rapid task switching.

 **3. <ins>Chunked prefill:<ins>**

  Chunked prefill allows large input prompts (prefills) to be divided into smaller chunks. These chunks can then be batched together with decode requests, which improves overall throughput and reduces latency during inference.

  - Process:
    - In traditional inference, the model processes all tokens sequentially, which can lead to inefficiencies, especially when dealing with long prompts.
    - With chunked prefill, vLLM can group multiple prefill tokens together and prioritize decode requests. This means that while the model is handling prefill requests, it can also process decode requests simultaneously.

  Benefits of Chunked Prefill

  - Improved Inter-Token Latency (ITL):
    By batching prefill and decode requests, chunked prefill reduces the time between generating tokens (inter-token latency). This is particularly beneficial in scenarios where quick responses are required.

  - Better GPU Utilization:
    The technique allows for more efficient use of GPU resources by overlapping compute-bound (prefill) and memory-bound (decode) operations. This leads to better overall performance and throughput.

  - Tuning Performance:
    - Users can adjust the `max_num_batched_tokens` parameter to optimize performance based on their specific workloads. The default value is set to 512, but increasing it can improve throughput at the cost of slightly higher latency for the first token generated.

  - Flexibility:
    - Chunked prefill can adapt to varying input sizes and workloads, making it suitable for applications with diverse requirements, such as summarization or question-answering tasks.

 **4. <ins>Guided Decoding:<ins>**

  Guided decoding refers to the process of constraining the output of a language model based on predefined rules or structures. This can include options such as predefined choices, regex patterns, JSON schemas, or grammars.
  The primary goal of guided decoding is to enhance the quality of generated outputs by steering the model towards specific formats or types of responses. This is particularly useful in applications where structured data is required, such as generating JSON responses or SQL queries.

  **5. <ins>Automatic Prefix Caching:<ins>**

  Automatic Prefix Caching (APC) allows the vLLM engine to cache the KV cache of existing queries. When a new query shares the same prefix as previous queries, it can reuse the cached KV data, significantly reducing the computational overhead associated with processing these queries. Useful in long document queries or multi-round conversations in chat applications

 **6. <ins>Prefill disaggregation:<ins>**

  Prefill disaggregation involves executing the prefill and decoding processes on separate resources (e.g., different GPUs). This separation allows each phase to be optimized independently, improving overall system performance

  Benefits of Disaggregation:
  - Reduced Interference: By separating the two phases, each can operate without hindering the other’s performance.
  - Tailored Resource Allocation: Resources can be allocated based on the specific needs of each phase, allowing for better optimization strategies.
  Improved Throughput: Disaggregating these phases can lead to higher goodput (the number of successful requests handled per unit time) as both phases can be scaled independently.

 **7. <ins>Pipeline Parallelism:<ins>**

Pipeline parallelism involves splitting a model into multiple stages, where each stage corresponds to a subset of the model's layers. Each stage can be assigned to different GPUs or nodes, allowing for concurrent processing of requests.

In a pipeline parallel setup, the model is divided into segments. For example, if a model has 28 layers and you have 4 GPUs, you might assign 7 layers to each GPU.
As data flows through the model, each GPU processes its assigned layers while passing intermediate results to the next GPU in the pipeline. This allows for overlapping computation and communication.

Pipeline parallelism can be combined with tensor parallelism to maximize resource utilization. Tensor parallelism splits the computations within a layer across multiple GPUs, while pipeline parallelism distributes entire layers across different GPUs.

Benefits of Pipeline Parallelism
- Increased Throughput:
By allowing multiple requests to be processed simultaneously across different stages of the model, pipeline parallelism can significantly increase throughput compared to traditional single-device setups.
- Reduced Latency for Batch Processing:
In scenarios where batch processing is common, pipeline parallelism helps minimize latency by keeping all GPUs busy and reducing idle time.
- Scalability:
This approach makes it feasible to work with larger models that exceed the memory capacity of individual GPUs by distributing them across multiple devices.
Optimized Resource Allocation:
Each GPU can be optimized for the specific layer it processes, allowing for better performance tuning based on the characteristics of each layer.

## Appendix-Kubernetes Deployment Explanation

Kubernetes Deployment YAML file in detail.

**template**

```yaml
template:
  metadata:
    labels:
      app: cancer-classification-server
      ai.gke.io/model: Llama-3.1-8B-Instruct
      ai.gke.io/inference-server: vllm
      examples.ai.gke.io/source: user-guide
```


  - **template**:
   The `template` field defines the pod template. This template is used to create the pods that are managed by the deployment.

   - **metadata**:
   The `metadata` field within the `template` specifies metadata for the pods. This includes labels that are applied to the pods.

   - **labels**:
   The `labels` field is a set of key-value pairs that are used to organize and select Kubernetes resources. Labels are used for various purposes, such as identifying and grouping resources, and for selecting resources using label selectors.

   - **app: cancer-classification-server**:
     This label indicates that the pod is part of the `cancer-classification-server` application. It is a common practice to use the `app` label to identify the application to which the pod belongs.

   - **ai.gke.io/model: Llama-3.1-8B-Instruct**:
     This label specifies the model being used by the pod. In this case, it is the `Llama-3.1-8B-Instruct` model. This label can be used to identify and manage pods that are running this specific model.

   - **ai.gke.io/inference-server: vllm**:
     This label indicates that the pod is using the `vllm` inference server. This label can be used to identify and manage pods that are running the vLLM inference server.

   - **examples.ai.gke.io/source: user-guide**:
     This label provides additional metadata about the source of the configuration. In this case, it indicates that the configuration is based on a user guide example. This label can be used for documentation or organizational purposes.


**spec**:

```yaml
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

- **spec**: Defines the specifications for the pod.
  - **containers**: Specifies the container configuration for the pod.
    - **name**: The name of the container (`inference-server`).
    - **image**: The Docker image to use for the container (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240930_0945_RC00`).
    - **resources**: Specifies the resource requests and limits for the container.
      - **requests**: The minimum amount of resources required for the container.
        - **cpu**: Requests 2 CPU cores.
        - **memory**: Requests 10 GiB of memory.
        - **ephemeral-storage**: Requests 10 GiB of ephemeral storage.
        - **nvidia.com/gpu**: Requests 1 NVIDIA GPU.
      - **limits**: The maximum amount of resources the container is allowed to use.
        - **cpu**: Limits the container to 2 CPU cores.
        - **memory**: Limits the container to 10 GiB of memory.
        - **ephemeral-storage**: Limits the container to 10 GiB of ephemeral storage.
        - **nvidia.com/gpu**: Limits the container to 1 NVIDIA GPU.



**command**

```yaml
command: ["python3", "-m", "vllm.entrypoints.api_server"]
```

- **command**: Specifies the command to run inside the container. This overrides the default command defined in the Docker image.
  - **"python3"**: The command to run the Python interpreter.
  - **"-m"**: A flag to run a module as a script.
  - **"vllm.entrypoints.api_server"**: The module to run. This is likely the entry point for the vLLM inference server.

**args**

```yaml
args:
- --model=$(MODEL_ID)
- --tensor-parallel-size=1
```

- **args**: Specifies the arguments to pass to the command. These arguments are passed to the `python3 -m vllm.entrypoints.api_server` command.
  - **--model=$(MODEL_ID)**: An argument specifying the model to be used by the inference server. The `$(MODEL_ID)` placeholder is replaced by the value of the `MODEL_ID` environment variable.
  - **--tensor-parallel-size=1**: An argument specifying the tensor parallel size, which is set to 1 in this case. Tensor parallelism is a technique used to distribute the computation of tensor operations across multiple devices.


**env**

```yaml
env:
- name: MODEL_ID
  value: Prat/Llama-3.1-8B-Instruct_ft_summarizer_v1
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-secret
      key: hf_api_token
```

- **env**: Specifies the environment variables to set inside the container.

  - **MODEL_ID**:
    - **name**: The name of the environment variable (`MODEL_ID`).
    - **value**: The value of the environment variable (`Prat/Llama-3.1-8B-Instruct_ft_summarizer_v1`).
    - This environment variable is used to specify the model ID that the inference server should use. The value `Prat/Llama-3.1-8B-Instruct_ft_summarizer_v1` is the identifier for the model.

  - **HUGGING_FACE_HUB_TOKEN**:
    - **name**: The name of the environment variable (`HUGGING_FACE_HUB_TOKEN`).
    - **valueFrom**: Specifies that the value of this environment variable should be sourced from a Kubernetes secret.
      - **secretKeyRef**: References a key within a Kubernetes secret.
        - **name**: The name of the secret (`hf-secret`).
        - **key**: The key within the secret (`hf_api_token`).
    - This environment variable is used to store the Hugging Face Hub token, which is necessary for authenticating with the Hugging Face API. The token is stored securely in a Kubernetes secret named `hf-secret`, and the specific key within the secret is `hf_api_token`.


**volumeMounts**

```yaml
volumeMounts:
- mountPath: /dev/shm
  name: dshm
```

- **volumeMounts**: Specifies the volumes to mount into the container's filesystem.

  - **mountPath**: The path inside the container where the volume will be mounted. In this case, the volume is mounted at `/dev/shm`.
    - **/dev/shm**: This is a special directory in Linux that is used for shared memory. Mounting a volume here can be useful for applications that require fast, temporary storage.

  - **name**: The name of the volume to mount. This name must match a volume defined in the `volumes` section of the pod specification.
    - **dshm**: The name of the volume to mount. This name should correspond to a volume defined elsewhere in the pod specification.


**nodeSelector**

```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-l4
  cloud.google.com/gke-gpu-driver-version: latest
```

- **nodeSelector**: Specifies the criteria for selecting the nodes on which the pods should be scheduled. It is a key-value map where each key-value pair represents a label that must be present on a node for it to be eligible to run the pod.

  - **cloud.google.com/gke-accelerator: nvidia-l4**:
    - **cloud.google.com/gke-accelerator**: This is a label key that is used to identify nodes with specific GPU accelerators.
    - **nvidia-l4**: This is the label value that specifies the type of GPU accelerator. In this case, it indicates that the node must have an NVIDIA L4 GPU.

  - **cloud.google.com/gke-gpu-driver-version: latest**:
    - **cloud.google.com/gke-gpu-driver-version**: This is a label key that is used to identify the GPU driver version installed on the node.
    - **latest**: This is the label value that specifies the GPU driver version. In this case, it indicates that the node must have the latest GPU driver version installed.


**apiVersion and kind**

```yaml
apiVersion: v1
kind: Service
```

- **apiVersion**: Specifies the API version for the service resource. `v1` is the current stable version for services.
- **kind**: Specifies the type of Kubernetes resource. In this case, it is a `Service`.

#### metadata

```yaml
metadata:
  name: llm-service
```

- **metadata**: Provides metadata for the service.
  - **name**: The name of the service. This name is used to identify the service within the Kubernetes cluster.

**spec**

```yaml
spec:
  selector:
    app: cancer-classification-server
  type: ClusterIP
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

- **spec**: Defines the desired state of the service, including the selector, type, and ports.

    - **selector**: Defines the label selector to identify the pods that the service will expose. The service will route traffic to the pods that match the specified labels.
      - **app: cancer-classification-server**: The label selector that matches pods with the label `app: cancer-classification-server`.

    - **type**: Specifies the type of service. `ClusterIP` is the default type, which exposes the service on a cluster-internal IP. This means the service is only accessible within the cluster.

    - **ports**: Defines the ports that the service will expose.
      - **protocol**: The protocol used by the service. In this case, it is `TCP`.
      - **port**: The port on which the service will be exposed. In this case, it is `8000`.
      - **targetPort**: The port on the pod to which traffic will be directed. In this case, it is `8000`.

