#!/bin/bash
python ./src/generate_data.py \
    --env Reach2D \
    --N_trajectories 1000 \
    --model_path ./out/dec25/ensemble_dagger_reach2d_mlp_test_refactor/model_4.pt \
    --seed 0 \
    --save_fname oracle_pi_r_mix_reach2d.pkl \
    --sample_mode oracle_pi_r_mix