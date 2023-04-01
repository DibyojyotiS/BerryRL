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

module --ignore-cache load python/conda-python/3.7
bash setup.sh -e base
source activate base
wandb login --relogin <api-key>
python ./agent-trainer/trainRunner.py
