#!/bin/bash
## Job Name
#SBATCH --job-name=mass_growth
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

sim_dir="/mnt/ceph/users/firesims/fire2/public_release/core/m12i_res7100/"
out_dir="/mnt/ceph/users/twagg/underworld/mass_growth/m12i_alt/"

# loop over redshifts from 3 to 0 in steps of 0.1
for redshift in $(seq 3.0 -0.1 0.0); do


    echo "Calculating dark matter mass growth at redshift z=${redshift}"

    python /mnt/home/twagg/projects/underworld/simulations/mass_growth_enclosed.py \
        --sim-dir "${sim_dir}" \
        --redshift "${redshift}" \
        --out-dir "${out_dir}"

    echo "Completed dark matter mass growth calculation at redshift z=${redshift}"
done