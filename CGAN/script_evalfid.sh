#!/bin/bash
#SBATCH --mail-type=ALL				 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=Anya-Aurore.Mauron@chuv.ch   # Where to send mail
#SBATCH --job-name=OCT_CGAN		 # Job name
#SBATCH --chdir=/home/an5770/CGAN		 # The working directory of the job
#SBATCH --account=rad
#SBATCH --partition=rad
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00				 # Total run time limit (HH:MM:SS)
#SBATCH --output=out/%N.%j.%a.out		 # Title of the output file
#SBATCH --error=err/%N.%j.%a.err		 # Title of the error file



loadanaconda
conda activate oct-patchbased-cgan  		 # activate the conda environment

srun python /home/an5770/CGAN/evalfid_70x70_patchbased.py 
echo "The node list is :\n$SLURM_NODELIST"