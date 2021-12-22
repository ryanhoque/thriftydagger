#!/bin/bash
python ./datasets/generate_data.py \
    --env Reach2D \
    --N_trajectories 1000 \
    --seed 0 \
    --save_fname scripted_oracle_reach2d.pkl \
    --sample_mode oracle