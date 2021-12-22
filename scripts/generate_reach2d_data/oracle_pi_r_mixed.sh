#!/bin/bash
python ./datasets/generate_data.py \
    --env Reach2D \
    --N_trajectories 1000 \
    --model_path ./out/12-22-2021-12:43:30/model_4.pt \
    --seed 0 \
    --save_fname oracle_pi_r_mix_reach2d.pkl \
    --sample_mode oracle_pi_r_mix