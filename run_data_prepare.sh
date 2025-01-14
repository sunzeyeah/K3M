#!/usr/bin/env bash

# activate conda environment, requirement: python==3.6, torch==1.4.0
#CONDA_ENV="py36_torch1.4"
#conda activate $CONDA_ENV

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
DATA_DIR=${ROOT_DIR}/raw
OUTPUT_DIR=${ROOT_DIR}/processed/k3m/
CV_MODEL_CONFIG="/root/autodl-tmp/Data/ccks2022/task9/output/cv_model/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml"
CV_MODEL_FILE="/root/autodl-tmp/Data/ccks2022/task9/output/cv_model/faster_rcnn_from_caffe.pkl"
MAIN="/root/Code/K3M/data_prepare.py"

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --file_item_info "item_info.jsonl" \
  --file_image "item_images" \
  --cv_model_config $CV_MODEL_CONFIG \
  --cv_model_file $CV_MODEL_FILE \
  --is_cuda

