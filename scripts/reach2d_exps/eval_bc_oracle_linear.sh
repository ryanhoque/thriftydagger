#!/bin/bash
python src/main.py \
    --exp_name dec25/test_bc_oracle_reach2d \
	--eval_only \
	--model_path ./out/dec25/test_bc_oracle_reach2d/model_4.pt\
    --environment Reach2D \
    --method BC \
    --arch LinearModel \
    --num_models 1 \
    --seed 4
