#!/bin/bash
## Job Name
#SBATCH --job-name=underworld
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=1:00:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_fiducial_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_fiducial_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting underworld simulation with file suffix: ${file_suffix}"

# run the distributed underworld simulation
python /mnt/home/twagg/projects/underworld/simulations/underworld.py \
    -n 20000000 \
    -o /mnt/ceph/users/twagg/underworld/sims/ \
    -p 64 \
    -s fiducial \
    -f "${file_suffix}"