#!/bin/bash
python src/main.py \
	--exp_name dec25/bc_oracle_reach2d \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch LinearModel \
    --num_models 1 \
    --seed 4 