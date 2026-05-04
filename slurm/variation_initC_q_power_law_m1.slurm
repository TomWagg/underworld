#!/bin/bash
## Job Name
#SBATCH --job-name=uw-q_power_law_m1
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:45:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_q_power_law_m1_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_q_power_law_m1_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting q_power_law_m1 underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/q_power_law_m1/q_power_law_m1${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/q_power_law_m1/q_power_law_m1${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld.py \
        -n 20000000 \
        -o /mnt/ceph/users/twagg/underworld/sims/ \
        -p 64 \
        -s "q_power_law_m1" \
        -f "${file_suffix}" \
        --q-power-law -0.999 --no-template
fi
