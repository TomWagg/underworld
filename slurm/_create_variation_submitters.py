from copy import copy

TEMPLATE = """#!/bin/bash
## Job Name
#SBATCH --job-name=uw-SIMNAMEREPLACE
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=0:30:00
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
    python /mnt/home/twagg/projects/underworld/simulations/underworld_from_template.py \\
        -o /mnt/ceph/users/twagg/underworld/sims/ \\
        -p 64 \\
        -s "SIMNAMEREPLACE" \\
        -f "${file_suffix}" \\
        --vary-params PARAMSREPLACE \\
        --reset-kicks
fi
"""

variations = []

# mt physics variations
for val in [0.0, 0.5, 1.0]:
    variations.append({"name": f"beta_{val}", "params": f"beta:{val}", "type": "mt"})
for val in [0.1, 0.5, 2.0, 10.0]:
    variations.append({"name": f"alpha_{val}", "params": f"alpha1:{val}", "type": "mt"})
for val in [0.001, 1000]:
    variations.append({"name": f"qcrit_caseB_{val}", "params": f"qcrit_2:{val}", "type": "mt"})

# remnant mass variations
variations.append({"name": "fryer_rapid", "params": "remnantflag:3", "type": "main"})
variations.append({"name": "mandel_muller", "params": "remnantflag:5 kickflag:6 mxns:2 rembar_massloss:0.1", "type": "main"})

for val in [0.0, 0.25, 0.50, 0.75, 1.0]:
    variations.append({
        "name": f"maltsev_fallback_{val}",
        "params": f"remnantflag:6 maltsev_mode:0 maltsev_fallback:{val} maltsev_pf_prob:0.1",
        "type": "main"
    })
for val in [0.0, 1.0]:
    variations.append({
        "name": f"maltsev_pf_prob_{val}",
        "params": f"remnantflag:6 maltsev_mode:0 maltsev_fallback:0.5 maltsev_pf_prob:{val}",
        "type": "main"
    })


# kick variations
variations.append({"name": "bhflag_0", "params": "bhflag:0", "type": "main"})
variations.append({"name": "bhflag_3", "params": "bhflag:3", "type": "main"})
variations.append({"name": "kickflag_1", "params": "kickflag:1", "type": "main"})

files = []

for variation in variations:
    submitter_script = copy(TEMPLATE)
    submitter_script = submitter_script.replace("SIMNAMEREPLACE", variation["name"])
    submitter_script = submitter_script.replace("PARAMSREPLACE", variation["params"])

    with open(f"variation_{variation['type']}_{variation['name']}.slurm", "w") as f:
        f.write(submitter_script)

    files.append(variation["name"])

print(files)
