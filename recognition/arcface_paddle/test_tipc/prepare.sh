#!/bin/bash
source test_tipc/common_func.sh
FILENAME=$1

# MODE be one of ['lite_train_infer' 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})
lines=(${dataline})

IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")
trainer_list=$(func_parser_value "${lines[14]}")

MODE=$2

if [ ${MODE} = "lite_train_lite_infer" ];then
    rm -rf MS1M_v2; mkdir MS1M_v2
    # pretrain lite train data
    tar xf test_tipc/data/small_dataset.tar --strip-components 1 -C MS1M_v2 
    
    # wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin
    cp test_tipc/data/small_lfw.bin MS1M_v2/lfw.bin

elif [ ${MODE} = "serving_infer" ];then
     # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    rm paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    wget https://paddle-serving.bj.bcebos.com/chain/paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install install paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install paddle_serving_client==0.6.3
    ${python_name} -m pip install paddle-serving-app==0.6.3
    ${python_name} -m pip install werkzeug==2.0.2

    rm -rf ./inference

    wget -nc -P ./inference https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar
    tar xf inference/mobileface_v1.0_infer.tar --strip-components 1 -C inference 
fi

