#! /bin/bash

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="utf-8"
export PATH="$HOME/local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/lib:$LD_LIBRARY_PATH"

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load gcc/7.4.0
module load cuda/10.2/10.2.89
module load cudnn/7.6/7.6.5
module load nccl/2.5/2.5.6-1

source venv/bin/activate

if [ -d $SERIALIZATION_DIR ]; then
    allennlp train $CONFIG_FILE \
    --serialization-dir $SERIALIZATION_DIR \
    --recover \
    --file-friendly-logging \
    --include-package modules
else
    allennlp train $CONFIG_FILE \
    --serialization-dir $SERIALIZATION_DIR \
    --file-friendly-logging \
    --include-package modules
fi
