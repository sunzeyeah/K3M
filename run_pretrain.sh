#!/usr/bin/env bash

# activate conda environment, requirement: python==3.6, torch==1.4.0
#CONDA_ENV="py36_torch1.4"
#conda activate $CONDA_ENV

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
DATA_DIR=${ROOT_DIR}/processed
OUTPUT_DIR=${ROOT_DIR}/output
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/bert/chinese_roberta_wwm_ext_pytorch"
MODEL_NAME="roberta_base"
MAIN="/root/Code/K3M/pretrain.py"
MAX_SEQ_LENGTH=50
MAX_SEQ_LENGTH_PV=256
MAX_NUM_PV=30
MAX_REGION_LENGTH=36
TRAIN_BATCH_SIZE=96
EVAL_BATCH_SIZE=96
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LOG_STEPS=10

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --file_name "{}_feat.lmdb" \
  --model_name $MODEL_NAME \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "k3m_roberta_base.json" \
  --pretrained_model_weights "roberta_base_weight_names.json" \
  --if_pre_sampling 1 \
  --with_coattention \
  --objective 0 \
  --visual_target 0 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --max_seq_length_pv $MAX_SEQ_LENGTH_PV \
  --max_num_pv $MAX_NUM_PV \
  --max_region_length $MAX_REGION_LENGTH \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --log_steps $LOG_STEPS \
  --fp16
