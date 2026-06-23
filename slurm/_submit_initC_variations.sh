#!/bin/bash

FIRST_PART=0
N_PARTS=34

# submit job arrays for each of the variation_*.slurm files in this directory
# wait 1 second between submissions to avoid overwhelming the scheduler

for slurm_file in variation_initC_*.slurm; do
    echo Submitting job array for $slurm_file with $N_PARTS parts...
    
    # submit the job array with the appropriate number of parts (0 to N_PARTS - 1)
    sbatch --array=$FIRST_PART-$((N_PARTS - 1)) "$slurm_file"

    # wait 1 second before submitting the next job array
    sleep 0.5
done