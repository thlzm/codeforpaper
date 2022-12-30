#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_classifier_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier_gpu.sh DEVICE_ID 1"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="$1"
fi


mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
#python ${PROJECT_DIR}/../run_classifier.py  \
python run_classifier.py  \
    --config_path="task_classifier_config.yaml" \
    --device_target="GPU" \
    --do_train="false" \
    --do_eval="true" \
    --assessment_method="Accuracy" \
    --epoch_num=10 \
    --num_class=2 \
    --train_data_shuffle="false" \
    --eval_data_shuffle="false" \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --save_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="" \
    --load_finetune_checkpoint_path="checkpoint/" \
    --train_data_file_path="" \
    --eval_data_file_path="data/" \
    --output_path=output/ \
    --schema_file_path="" > classifier_log.txt 2>&1 &
