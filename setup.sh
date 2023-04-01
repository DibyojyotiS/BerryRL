#!/bin/sh

if conda info --envs | grep -q ML; then echo "base already exists"; else conda create -y -n ML; fi
conda activate ML
pip install -r requirements.txt
pip install -e berry-field
cd ..
pip install -e DRLagents
cd ./BerryRL