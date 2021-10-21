#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer',  'infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2_small.tar
    tar xf MS1M_v2/MS1M_v2_small.tar --strip-components 1 -C MS1M_v2 
elif [ ${MODE} = "whole_train_infer" ];then
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2.tar
    tar xf MS1M_v2/MS1M_v2.tar --strip-components 1 -C MS1M_v2 

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin
elif [ ${MODE} = "whole_infer" ];then
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2_small.tar
    tar xf MS1M_v2/MS1M_v2_small.tar --strip-components 1 -C MS1M_v2 

    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/lfw.bin
elif [ ${MODE} = "infer" ];then
    rm -rf ./MS1M_v2
    rm -rf ./inference
    rm -rf ./MS1M_v2
    wget -nc -P ./MS1M_v2/ https://paddle-model-ecology.bj.bcebos.com/whole_chain/insight-face/MS1M_v2_small.tar
    tar xf MS1M_v2/MS1M_v2_small.tar --strip-components 1 -C MS1M_v2 
    
    wget -nc -P ./inference https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar
    tar xf inference/mobileface_v1.0_infer.tar --strip-components 1 -C inference 
    mv inference/inference.pdiparams inference/${model_name}.pdiparams
    mv inference/inference.pdiparams.info inference/${model_name}.pdiparams.info
    mv inference/inference.pdmodel inference/${model_name}.pdmodel
fi
