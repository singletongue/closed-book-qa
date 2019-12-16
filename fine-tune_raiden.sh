#!/bin/bash
#$ -jc gpu-container_g1
#$ -ac d=nvcr-pytorch-1910
#$ -cwd
#$ -j y

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

source $HOME/.raiden/setup_gpu.sh
source /fefs/opt/dgx/env_set/nvcr-pytorch-1910.sh
source python/bin/activate

allennlp fine-tune \
--model-archive $MODEL_ARCHIVE \
--config-file $CONFIG_FILE \
--serialization-dir $SERIALIZATION_DIR \
--file-friendly-logging \
--include-package modules
