#!/usr/bin/env bash

# activate conda environment, requirement: python==3.6, torch==1.4.0
#CONDA_ENV="py36_torch1.4"
#conda activate $CONDA_ENV

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
DATA_DIR=${ROOT_DIR}/processed
OUTPUT_DIR=${ROOT_DIR}/output
PRETRAINED_BERT_PATH="/root/autodl-tmp/Data/bert/chinese_roberta_wwm_ext_pytorch"
PRETRAINED_MODEL_PATH="${OUTPUT_DIR}/k3m_roberta_base_12l_12h/K3M_struc_presample-1_epoch-4.bin"
MODEL_NAME="roberta_base"
MAIN="/root/Code/K3M/finetune.py"
MAX_SEQ_LENGTH=50
MAX_SEQ_LENGTH_PV=256
MAX_NUM_PV=30
MAX_REGION_LENGTH=36
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LOG_STEPS=10

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --file_name "train_{}_item_alignment.lmdb" \
  --model_name $MODEL_NAME \
  --config_file "k3m_roberta_base.json" \
  --pretrained_model_path $PRETRAINED_BERT_PATH \
  --file_state_dict $PRETRAINED_MODEL_PATH \
  --do_eval \
  --if_pre_sampling 1 \
  --with_coattention \
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
