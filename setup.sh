#!/bin/sh

## accepts envname as a flag -e
while getopts e: flag
do
    case "${flag}" in
        e) envname=${OPTARG};;
    esac
done

if conda info --envs | grep -q $envname; then echo "$envname already exists"; else conda create -n $envname python=3.7.6; fi

echo "doing conda init bash"
conda init bash

echo "trying to activate $envname"
source activate $envname
conda env list
conda env list | grep "$envname * [\*]"
if conda env list | grep "$envname * [\*]"; then echo "Created env $envname"; else echo "failed somewhere"; exit; fi

echo "installing stuff"
pip install -r requirements.txt
pip install -e berry-field
cd ..
pip install -e DRLagents
cd ./BerryRL