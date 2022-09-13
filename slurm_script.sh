#!/bin/sh
#SBATCH -N 16 // specifies number of nodes
#SBATCH --ntasks-per-node=40 // specifies core per node
#SBATCH --time=06:50:20 // specifies maximum duration of run
#SBATCH --job-name=lammps // specifies job name
#SBATCH --error=job.%J.err_node_40 // specifies error file name
#SBATCH --output=job.%J.out_node_40 //specifies output file name
#SBATCH --partition=gpu // specifies queue name
export I_MPI_FABRIC

module load python/conda-python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
