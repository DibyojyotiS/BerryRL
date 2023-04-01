#!/bin/sh

if conda info --envs | grep -q ML; then echo "base already exists"; else conda create -y -n ML; fi
conda init

conda activate ML
cond env list
if conda info --envs | grep -q ML; then echo "Created env ML"; else echo "failed somewhere"; exit; fi

pip install -r requirements.txt
pip install -e berry-field
cd ..
pip install -e DRLagents
cd ./BerryRL