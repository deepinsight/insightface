
# Service deployment based on PaddleServing  

(English|[简体中文](./README_CN.md))


This document will introduce how to use the [PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README.md) to deploy the Arcface dynamic graph model as a pipeline online service.

Some Key Features of Paddle Serving:
- Integrate with Paddle training pipeline seamlessly, most paddle models can be deployed with one line command.
- Industrial serving features supported, such as models management, online loading, online A/B testing etc.
- Highly concurrent and efficient communication between clients and servers supported.

The introduction and tutorial of Paddle Serving service deployment framework reference [document](https://github.com/PaddlePaddle/Serving/blob/develop/README.md).


## Contents
- [Environmental preparation](#environmental-preparation)
- [Model conversion](#model-conversion)
- [Paddle Serving pipeline deployment](#paddle-serving-pipeline-deployment)
- [FAQ](#faq)

<a name="environmental-preparation"></a>
## Environmental preparation

Arcface operating environment and Paddle Serving operating environment are needed.

1. Please prepare Arcface operating environment reference [link](../../README_en.md).
   Download the corresponding paddle whl package according to the environment, it is recommended to install version 2.2+.


2. The steps of PaddleServing operating environment prepare are as follows:

    Install serving which used to start the service
    ```
    pip3 install paddle-serving-server==0.6.3 # for CPU
    pip3 install paddle-serving-server-gpu==0.6.3 # for GPU
    # Other GPU environments need to confirm the environment and then choose to execute the following commands
    pip3 install paddle-serving-server-gpu==0.6.3.post101 # GPU with CUDA10.1 + TensorRT6
    pip3 install paddle-serving-server-gpu==0.6.3.post11 # GPU with CUDA11 + TensorRT7
    ```

3. Install the client to send requests to the service
    In [download link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md) find the client installation package corresponding to the python version.
    The python3.7 version is recommended here:

    ```
    pip3 install paddle-serving-client==0.6.3
    ```

4. Install serving-app
    ```
    pip3 install paddle-serving-app==0.6.3
    ```

   **note:** If you want to install the latest version of PaddleServing, refer to [link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md).


<a name="model-conversion"></a>
## Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a serving model that is easy to deploy.

Firstly, download the inference model of Arcface
```
wget -nc -P ./inference https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar
tar xf inference/mobileface_v1.0_infer.tar --strip-components 1 -C inference 
```
Then, you can use installed paddle_serving_client tool to convert inference model to mobile model.
```
python3 -m paddle_serving_client.convert --dirname ./inference/ \
                                         --model_filename inference.pdmodel \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./MobileFaceNet_128_serving/ \
                                         --serving_client ./MobileFaceNet_128_client/

```

After the detection model is converted, there will be additional folders of `MobileFaceNet_128_serving` and `MobileFaceNet_128_client` in the current folder, with the following format:
```
MobileFaceNet_128_serving
├── __model__
├── __params__
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt

MobileFaceNet_128_client/
├── serving_client_conf.prototxt
└── serving_client_conf.stream.prototxt

```
The recognition model is the same.

<a name="paddle-serving-pipeline-deployment"></a>
## Paddle Serving pipeline deployment

1. Download the PaddleOCR code, if you have already downloaded it, you can skip this step.
    ```
    git clone https://github.com/deepinsight/insightface

    # Enter the working directory  
    cd recognition/arcface_paddle/deploy/pdserving
    ```

    The pdserver directory contains the code to start the pipeline service and send prediction requests, including:
    ```
    __init__.py
    config.yml # Start the service configuration file
    ocr_reader.py # pre-processing and post-processing code implementation
    pipeline_http_client.py # Script to send pipeline prediction request
    web_service.py # Start the script of the pipeline server
    ```

2. Run the following command to start the service.
    ```
    # Start the service and save the running log in log.txt
    python3 web_service.py &>log.txt &
    ```
    After the service is successfully started, a log similar to the following will be printed in log.txt
    ![](./imgs/start_server.png)

3. Send service request
    ```
    python3 pipeline_http_client.py
    ```
    After successfully running, the predicted result of the model will be printed in the cmd window. An example of the result is:
    ![](./imgs/results.png)  

    Adjust the number of concurrency in config.yml to get the largest QPS. Generally, the number of concurrent detection and recognition is 2:1

    ```
    det:
        concurrency: 8
        ...
    rec:
        concurrency: 4
        ...
    ```

    Multiple service requests can be sent at the same time if necessary.

    The predicted performance data will be automatically written into the `PipelineServingLogs/pipeline.tracer` file.

    Tested on 700 real picture. The average QPS on V100 GPU can reach around 57:

    ```
    2021-11-04 13:38:52,507 Op(ArcFace):
    2021-11-04 13:38:52,507 	in[135.4579597902098 ms]
    2021-11-04 13:38:52,507 	prep[0.9921311188811189 ms]
    2021-11-04 13:38:52,507 	midp[3.9232132867132865 ms]
    2021-11-04 13:38:52,507 	postp[0.12166258741258741 ms]
    2021-11-04 13:38:52,507 	out[0.9898286713286714 ms]
    2021-11-04 13:38:52,508 	idle[0.9643989520087675]
    2021-11-04 13:38:52,508 DAGExecutor:
    2021-11-04 13:38:52,508 	Query count[573]
    2021-11-04 13:38:52,508 	QPS[57.3 q/s]
    2021-11-04 13:38:52,509 	Succ[0.9982547993019197]
    2021-11-04 13:38:52,509 	Error req[394]
    2021-11-04 13:38:52,509 	Latency:
    2021-11-04 13:38:52,509 		ave[11.52941186736475 ms]
    2021-11-04 13:38:52,509 		.50[11.492 ms]
    2021-11-04 13:38:52,509 		.60[11.658 ms]
    2021-11-04 13:38:52,509 		.70[11.95 ms]
    2021-11-04 13:38:52,509 		.80[12.251 ms]
    2021-11-04 13:38:52,509 		.90[12.736 ms]
    2021-11-04 13:38:52,509 		.95[13.21 ms]
    2021-11-04 13:38:52,509 		.99[13.987 ms]
    2021-11-04 13:38:52,510 Channel (server worker num[10]):
    2021-11-04 13:38:52,510 	chl0(In: ['@DAGExecutor'], Out: ['ArcFace']) size[0/0]
    2021-11-04 13:38:52,510 	chl1(In: ['ArcFace'], Out: ['@DAGExecutor']) size[0/0]
    ```

<a name="faq"></a>
## FAQ
**Q1**: No result return after sending the request.

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and before sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```  
