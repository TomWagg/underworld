#!/bin/bash
## Job Name
#SBATCH --job-name=m12i_evol
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:10:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/m12i_evol_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/m12i_evol_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

sim_dir="/mnt/ceph/users/firesims/fire2/public_release/core/m12i_res7100/"
out_dir="/mnt/ceph/users/twagg/underworld/mass_growth/m12i"

# create an array of redshifts from 3.0 to 0.0 in steps of -0.1
# then pick the redshift based on SLURM_ARRAY_TASK_ID
redshifts=($(seq 3.0 -0.1 0.0))
redshift=${redshifts[$SLURM_ARRAY_TASK_ID]}

# repeat for two circularity thresholds, 0.0 and 0.5
for circularity_thresh in 0.0 0.5; do
    out_dir_circularity="${out_dir}/eps_${circularity_thresh}/"
    mkdir -p "${out_dir_circularity}"

    echo "Calculating dark matter mass growth at redshift z=${redshift}, with eps=${circularity_thresh}"

    python /mnt/home/twagg/projects/underworld/simulations/mass_growth_enclosed.py \
        --sim-dir "${sim_dir}" \
        --redshift "${redshift}" \
        --out-dir "${out_dir_circularity}" \
        --circularity-thresh "${circularity_thresh}" \
        --overwrite
done

echo "Completed dark matter mass growth calculation at redshift z=${redshift}"