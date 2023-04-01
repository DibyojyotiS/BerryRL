#!/bin/sh

if conda info --envs | grep -q ML; then echo "base already exists"; else conda create -n ML python=3.7.6; fi
conda init

conda activate ML
conda env list
if conda env list | grep "ML * [\*]"; then echo "Created env ML"; else echo "failed somewhere"; exit; fi

pip install -r requirements.txt
pip install -e berry-field
cd ..
pip install -e DRLagents
cd ./BerryRL