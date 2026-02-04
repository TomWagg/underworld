#!/bin/bash
## Job Name
#SBATCH --job-name=underworld
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:45:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

for alpha in 0.0 0.1 0.5 2.0 10.0; do

    # skip this item if file is already present
    if [ -f "/mnt/ceph/users/twagg/underworld/alpha-${alpha}/alpha-${alpha}${file_suffix}.h5" ]; then
        echo "File /mnt/ceph/users/twagg/underworld/alpha-${alpha}/alpha-${alpha}${file_suffix}.h5 already exists, skipping..."
        continue
    fi

    echo "Starting underworld simulation with file suffix: ${file_suffix}, varying alpha to ${alpha}"

    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/distributed_underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/alpha-${alpha}/ \
        -p 64 \
        -s alpha-${alpha} \
        -f "${file_suffix}" \
        --vary-params qcflag:5 alpha1:${alpha} \
        --reset-kicks

done