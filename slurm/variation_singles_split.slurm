#!/bin/bash
## Job Name
#SBATCH --job-name=singles-split
#SBATCH --partition=cca,gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_singles_split_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_singles_split_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting singles underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/sims/singsdfsdles_split/singles_split${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/sims/singles_split/singles_split${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/sims/ \
        -p 64 \
        -s "singles_split" \
        -f "${file_suffix}" \
        --reset-kicks \
        --run-as-singles \
        --singles-method split
fi
