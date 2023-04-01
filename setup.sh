#!/bin/sh

if conda info --envs | grep -q ML; then echo "ML already exists"; else conda create -n ML python=3.7.6; fi

echo "doing conda init bash"
conda init bash

echo "trying to activate ML"
conda activate ML
conda env list
conda env list | grep "ML * [\*]"
if conda env list | grep "ML * [\*]"; then echo "Created env ML"; else echo "failed somewhere"; exit; fi

echo "installing stuff"
pip install -r requirements.txt
pip install -e berry-field
cd ..
pip install -e DRLagents
cd ./BerryRL