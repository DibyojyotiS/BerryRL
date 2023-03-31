#!/bin/sh
#SBATCH -N 1 / Number of nodes
#SBATCH --ntasks-per-node=1 / Number of cores for node
#SBATCH --time=2-00:00:00 / Time required to execute the program
#SBATCH --job-name=berryRL / Name of application
#SBATCH --error=job.%J.err / Name of the output file
#SBATCH --output=job.%J.out / Name of the error file
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu / Partition or queue name 
#SBATCH --mail-user=dibyo@iitk.ac.in

module load conda-python/3.7
source activate base

# TODO Install requirements
# install DRLagents and berry-env

python trainRunner.py