#!/bin/bash
python ./src/generate_data.py \
    --environment Reach2D \
    --N_trajectories 1000 \
    --model_path ./out/dec25/ensemble_dagger_reach2d_mlp_test_refactor/model_4.pt \
    --seed 0 \
    --save_fname pi_r.pkl \
    --sample_mode pi_r