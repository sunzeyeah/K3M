#!/usr/bin/env bash

# activate conda environment, requirement: python==3.6, torch==1.4.0
#CONDA_ENV="py36_torch1.4"
#conda activate $CONDA_ENV

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
DATA_DIR=${ROOT_DIR}/processed/k3m
OUTPUT_DIR=${ROOT_DIR}/output
PRETRAINED_BERT_PATH="/root/autodl-tmp/Data/bert/roberta_base"
MODEL_NAME="k3m_base"
MAIN="/root/Code/K3M/finetune.py"
THRESHOLD=0.5
EPOCH=4
PRETRAINED_MODEL_PATH="${OUTPUT_DIR}/item_alignment_k3m_base/K3M_item_alignment-1_epoch-${EPOCH}.bin"
EVAL_BATCH_SIZE=1024

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --file_name "{}_item_alignment.lmdb" \
  --model_name $MODEL_NAME \
  --config_file "k3m_roberta_base.json" \
  --pretrained_model_path $PRETRAINED_BERT_PATH \
  --file_state_dict $PRETRAINED_MODEL_PATH \
  --do_pred \
  --use_image \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --threshold $THRESHOLD \
  --log_steps 10 \
  --if_pre_sampling 1 \
  --with_coattention \
  --visual_target 0 \
  --max_seq_length 50 \
  --max_seq_length_pv 256 \
  --max_num_pv 30 \
  --max_region_length 36 \
  --fp16
