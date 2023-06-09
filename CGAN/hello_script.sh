#!/bin/bash
#SBATCH --mail-type=ALL				 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=Anya-Aurore.Mauron@chuv.ch   # Where to send mail
#SBATCH --job-name=OCT_CGAN_ 			 # Job name
#SBATCH --chdir=/home/an5770/CGAN		 # The working directory of the job
#SBATCH --account=rad
#SBATCH --partition=rad
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00				 # Total run time limit (HH:MM:SS)
#SBATCH --output=out/out%N.%j.%a.out		 # Title of the output file
#SBATCH --error=err/err%N.%j.%a.err		 # Title of the error file


srun python hello.py
