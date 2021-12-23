#!/bin/bash
python main.py \
	--exp_name dec23_oracle_reach2d_mlp_no_tanh \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch MLP \
	--hidden_size 20 \
    --num_models 1 \
    --seed 4
