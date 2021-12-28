#!/bin/bash
ENVIRONMENT=Reach2D
N=1000
SAMPLE_MODE=pi_r
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $SAMPLE_MODE.pkl \
    --sample_mode $SAMPLE_MODE