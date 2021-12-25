#!/bin/bash
python src/main.py \
	--exp_name dec24/oracle_reach2d_mlp_test_refactor \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch MLP \
	--hidden_size 20 \
    --num_models 1 \
    --seed 4 \
	--overwrite
