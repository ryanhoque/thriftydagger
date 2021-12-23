#!/bin/bash
python main.py \
	--eval_only \
	--model_path ./out/dec22_bc_oracle_reach2d_linear/model_4.pt\
    --environment Reach2D \
    --method BC \
    --arch LinearModel \
    --num_models 1 \
    --seed 4
