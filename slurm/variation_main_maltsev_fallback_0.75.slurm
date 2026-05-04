#!/bin/bash
## Job Name
#SBATCH --job-name=uw-maltsev_fallback_0.75
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_maltsev_fallback_0.75_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_maltsev_fallback_0.75_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting maltsev_fallback_0.75 underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/maltsev_fallback_0.75/maltsev_fallback_0.75${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/maltsev_fallback_0.75/maltsev_fallback_0.75${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/sims/ \
        -p 64 \
        -s "maltsev_fallback_0.75" \
        -f "${file_suffix}" \
        --vary-params remnantflag:6 maltsev_mode:0 maltsev_fallback:0.75 maltsev_pf_prob:0.1 \
        --reset-kicks
fi
