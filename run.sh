#!/bin/bash
#SBATCH --job-name=noscale
#SBATCH --output=both.x
#SBATCH --error=both.x
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 
#SBATCH --partition=gpu

echo "SLURM_NTASKS: $SLURM_NTASKS"

# source ~/.bash_profile

if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
	echo "Error: SLURM_NTASKS is not set or is less than or equal to 0"
	exit 1
fi

amw25 --cwd tot --mode tot
