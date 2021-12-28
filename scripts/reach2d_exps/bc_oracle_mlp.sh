#!/bin/bash
python src/main.py \
	--exp_name dec25/oracle_reach2d_mlp \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch MLP \
	--hidden_size 20 \
    --num_models 1 \
    --seed 4
