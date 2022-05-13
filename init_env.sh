# create conda virtual environment
conda create -n py38_torch1.6 python=3.8
conda activate py38_torch1.6

# python packages
pip install -r requirements.txt

# install torch
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
