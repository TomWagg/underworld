#!/bin/bash
## Job Name
#SBATCH --job-name=underworld
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=1:00:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting underworld simulation with file suffix: ${file_suffix}, varying to Maltsev model"

# loop over fallback: 0, 0.25, 0.5, 0.75, 1.0
for fallback in 0 0.25 0.5 0.75 1.0; do

    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/distributed_underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/ \
        -p 64 \
        -s "maltsev-fallback-${fallback}" \
        -f "${file_suffix}" \
        --vary-params qcflag:5 remnantflag:6 maltsev_mode:0 maltsev_fallback:${fallback} \
        --reset-kicks
done