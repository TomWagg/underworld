#!/bin/bash

N_PARTS=5

for slurm_file in variation_main_*.slurm; do
    echo Submitting job array for $slurm_file with $MAX_PARTS parts...
    
    # # submit the job array with the appropriate number of parts
    sbatch --array=0-$((N_PARTS - 1)) "$slurm_file"

    # wait 1 second before submitting the next job array
    sleep 0.5
done