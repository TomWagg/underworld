#!/bin/bash
## Job Name
#SBATCH --job-name=uw-imf_2.7
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:45:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_imf_2.7_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_imf_2.7_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting imf_2.7 underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/imf_2.7/imf_2.7${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/imf_2.7/imf_2.7${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld.py \
        -n 20000000 \
        -o /mnt/ceph/users/twagg/underworld/sims/ \
        -p 64 \
        -s "imf_2.7" \
        -f "${file_suffix}" \
        -m -2.7 --no-template
fi
