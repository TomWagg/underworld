#!/bin/bash
## Job Name
#SBATCH --job-name=underworld
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

for beta in 0.0 0.5 1.0; do

    echo "Starting underworld simulation with file suffix: ${file_suffix}, varying beta to ${beta}"

    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/distributed_underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/beta-${beta}/ \
        -p 64 \
        -s beta-${beta} \
        -f "${file_suffix}" \
        --vary-params qcflag:5 acc_lim:${beta} \
        --reset-kicks

done