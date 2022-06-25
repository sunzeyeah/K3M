#!/usr/bin/env bash

# activate conda environment, requirement: python==3.6, torch==1.4.0
#CONDA_ENV="py36_torch1.4"
#conda activate $CONDA_ENV

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
DATA_DIR=${ROOT_DIR}/processed/k3m
OUTPUT_DIR=${ROOT_DIR}/output
PRETRAINED_BERT_PATH="/root/autodl-tmp/Data/bert/roberta_base"
PRETRAINED_MODEL_PATH="${OUTPUT_DIR}/k3m_roberta_base_12l_12h/K3M_struc_presample-1_epoch-4.bin"
#PRETRAINED_MODEL_PATH="${OUTPUT_DIR}/k3m_item_alignment_roberta_base_12l_12h/K3M_item_alignment-1_epoch-0.bin"
MODEL_NAME="k3m_base"
MAIN="/root/Code/K3M/finetune.py"
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
LEARNING_RATE=5e-5

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --file_name "{}_item_alignment.lmdb" \
  --model_name $MODEL_NAME \
  --config_file "k3m_roberta_base.json" \
  --pretrained_model_path $PRETRAINED_BERT_PATH \
  --file_state_dict $PRETRAINED_MODEL_PATH \
  --use_image \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --start_epoch 0 \
  --num_train_epochs 10 \
  --log_steps 10 \
  --warmup_proportion 0.3 \
  --if_pre_sampling 1 \
  --with_coattention \
  --visual_target 0 \
  --max_seq_length 50 \
  --max_seq_length_pv 256 \
  --max_num_pv 30 \
  --max_region_length 36 \
  --fp16
