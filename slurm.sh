#!/bin/sh
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=berryRL
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-user=dibyo@iitk.ac.in

module --ignore-cache load conda-python/3.7

source activate RL
conda env list

echo "here1"

python train.py
## TODO Install requirements
## install DRLagents and berry-env
## conda install -c conda-forge ffmpeg
## python trainRunner.py