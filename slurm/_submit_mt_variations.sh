#!/bin/bash

MAX_PARTS=33

# submit job arrays for each of the variation_*.slurm files in this directory
# wait 1 second between submissions to avoid overwhelming the scheduler

for slurm_file in variation_*.slurm; do
    # skip any file that doesn't include "alpha", "beta", or "qcrit" in the name
    if [[ "$slurm_file" != *"alpha"* && "$slurm_file" != *"beta"* && "$slurm_file" != *"qcrit"* ]]; then
        continue
    fi
    echo Submitting job array for $slurm_file with $MAX_PARTS parts...
    
    # # submit the job array with the appropriate number of parts
    sbatch --array=0-$MAX_PARTS "$slurm_file"

    # wait 1 second before submitting the next job array
    sleep 0.5
done