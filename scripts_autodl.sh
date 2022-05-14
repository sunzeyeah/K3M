#!/usr/bin/env bash

ROOT_DIR=root
mkdir /$ROOT_DIR/Code

# clone code
cd /$ROOT_DIR/Code
git clone https://github.com/sunzeyeah/K3M.git

# install packages
cd K3M
pip install -r requirements.txt
pip uninstall tensorboard



