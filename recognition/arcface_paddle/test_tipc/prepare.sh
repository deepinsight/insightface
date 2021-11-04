#!/bin/bash
source test_tipc/common_func.sh
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})
lines=(${dataline})

IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")
trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2_small.tar
    tar xf MS1M_v2/MS1M_v2_small.tar --strip-components 1 -C MS1M_v2 
    
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lite_infer.tar

    cd MS1M_v2;tar xf lite_infer.tar; cd ..
elif [ ${MODE} = "whole_train_whole_infer" ];then
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2.tar
    tar xf MS1M_v2/MS1M_v2.tar --strip-components 1 -C MS1M_v2 

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lite_infer.tar

    cd MS1M_v2;tar xf lite_infer.tar; cd ..
elif [ ${MODE} = "lite_train_whole_infer" ];then
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2_small.tar
    tar xf MS1M_v2/MS1M_v2_small.tar --strip-components 1 -C MS1M_v2 

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lite_infer.tar

    cd MS1M_v2;tar xf lite_infer.tar; cd ..
elif [ ${MODE} = "whole_infer" ];then
    rm -rf ./MS1M_v2
    rm -rf ./inference
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lite_infer.tar

    cd MS1M_v2;tar xf lite_infer.tar; cd ..
    
    wget -nc -P ./inference https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar
    tar xf inference/mobileface_v1.0_infer.tar --strip-components 1 -C inference 
    mv inference/inference.pdiparams inference/${model_name}.pdiparams
    mv inference/inference.pdiparams.info inference/${model_name}.pdiparams.info
    mv inference/inference.pdmodel inference/${model_name}.pdmodel
elif [ ${MODE} = "serving_infer" ];then
     # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    rm paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    wget https://paddle-serving.bj.bcebos.com/chain/paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install install paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install paddle_serving_client==0.6.3
    ${python_name} -m pip install paddle-serving-app==0.6.3
    ${python_name} -m pip install -U werkzeug

    rm -rf ./inference

    wget -nc -P ./inference https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar
    tar xf inference/mobileface_v1.0_infer.tar --strip-components 1 -C inference 
fi

