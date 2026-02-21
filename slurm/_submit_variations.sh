#!/bin/bash

MAX_PARTS=5

# submit job arrays for each of the variation_*.slurm files in this directory
# wait 1 second between submissions to avoid overwhelming the scheduler

for slurm_file in variation_*.slurm; do
    echo Submitting job array for $slurm_file with $MAX_PARTS parts...
    
    # # submit the job array with the appropriate number of parts
    sbatch --array=0-$MAX_PARTS "$slurm_file"

    # wait 1 second before submitting the next job array
    sleep 0.5
done