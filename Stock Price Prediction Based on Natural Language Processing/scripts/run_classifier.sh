#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
echo "Please run the scipt as: "
echo "bash scripts/run_classifier.sh"
echo "for example: bash scripts/run_classifier.sh"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

sudo mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
sudo python -u ${PROJECT_DIR}/../my_ubuntu_classifier.py  \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --do_predict="true" \
    --assessment_method="f1" \
    --device_id=0 \
    --epoch_num=4 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --predict_data_shuffle="false" \
    --train_batch_size=128 \
    --eval_batch_size=8 \
    --predict_batch_size=8 \
    --save_finetune_checkpoint_path="./data/base" \
    --load_pretrain_checkpoint_path="./data/base/bert_base.ckpt" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="./data/train.tf_record" \
    --eval_data_file_path="./data/eval.tf_record" \
    --predict_data_file_path="./data/predict.tf_record" \
    --predict_result_path='./data/predict_result.csv' \
    --schema_file_path=""  2>&1 | tee -a classifier_log.txt &
#> classifier_log.txt 2>&1 &