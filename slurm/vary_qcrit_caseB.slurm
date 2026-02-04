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

for qcrit_caseB in 0.001 1000; do

    # skip this item if file is already present
    if [ -f "/mnt/ceph/users/twagg/underworld/qcrit-caseB-${qcrit_caseB}/qcrit-caseB-${qcrit_caseB}${file_suffix}.h5" ]; then
        echo "File /mnt/ceph/users/twagg/underworld/qcrit-caseB-${qcrit_caseB}/qcrit-caseB-${qcrit_caseB}${file_suffix}.h5 already exists, skipping..."
        continue
    fi

    echo "Starting underworld simulation with file suffix: ${file_suffix}, varying qcrit_caseB to ${qcrit_caseB}"

    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/distributed_underworld_from_template.py \
        -o /mnt/ceph/users/twagg/underworld/qcrit-caseB-${qcrit_caseB}/ \
        -p 64 \
        -s qcrit-caseB-${qcrit_caseB} \
        -f "${file_suffix}" \
        --vary-params qcflag:5 qcrit_2:${qcrit_caseB} \
        --reset-kicks

done