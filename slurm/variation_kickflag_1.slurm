#!/bin/bash
## Job Name
#SBATCH --job-name=uw-kickflag_1
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_kickflag_1_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_kickflag_1_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting kickflag_1 underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/kickflag_1/kickflag_1${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/kickflag_1/kickflag_1${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/ \
        -p 64 \
        -s "kickflag_1" \
        -f "${file_suffix}" \
        --vary-params kickflag:1 \
        --reset-kicks
fi
