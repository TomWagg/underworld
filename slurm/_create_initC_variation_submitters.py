from copy import copy

TEMPLATE = """#!/bin/bash
## Job Name
#SBATCH --job-name=uw-SIMNAMEREPLACE
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:45:00
#SBATCH -o /mnt/home/twagg/projects/underworld/logs/logs_SIMNAMEREPLACE_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/underworld/logs/logs_SIMNAMEREPLACE_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# determine the file suffix based on SLURM_ARRAY_TASK_ID
file_suffix="_part${SLURM_ARRAY_TASK_ID}"

echo "Starting SIMNAMEREPLACE underworld simulation with file suffix: ${file_suffix}"

if [ -f "/mnt/ceph/users/twagg/underworld/SIMNAMEREPLACE/SIMNAMEREPLACE${file_suffix}.h5" ]; then
    echo "File /mnt/ceph/users/twagg/underworld/SIMNAMEREPLACE/SIMNAMEREPLACE${file_suffix}.h5 already exists, skipping..."
else
    # run the distributed underworld simulation
    python /mnt/home/twagg/projects/underworld/simulations/underworld.py \\
        -n 20000000 \\
        -o /mnt/ceph/users/twagg/underworld/sims/ \\
        -p 64 \\
        -s "SIMNAMEREPLACE" \\
        -f "${file_suffix}" \\
        OPTIONSREPLACE
fi
"""

variations = [
    {"name": "imf_1.9", "options": "-m -1.9"},
    {"name": "imf_2.7", "options": "-m -2.7"},
    {"name": "porb_slope_0", "options": "--porb-model 0.0"},
    {"name": "porb_slope_m1", "options": "--porb-model -0.999"},
    {"name": "porb_max_1000", "options": "--porb-max 3.0"},
    {"name": "q_power_law_m1", "options": "--q-power-law -0.999"},
    {"name": "q_power_law_p1", "options": "--q-power-law 1.0"},
    {"name": "qmin_pre_ms", "options": "--qmin -1 --m2-min -1"},
]

files = []

for variation in variations:
    submitter_script = copy(TEMPLATE)
    submitter_script = submitter_script.replace("SIMNAMEREPLACE", variation["name"])
    submitter_script = submitter_script.replace("OPTIONSREPLACE", variation["options"] + " --no-template")

    with open(f"variation_initC_{variation['name']}.slurm", "w") as f:
        f.write(submitter_script)

    files.append(variation["name"])

print(files)
