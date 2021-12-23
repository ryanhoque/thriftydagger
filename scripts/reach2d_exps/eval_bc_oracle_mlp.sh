#!/bin/bash
python main.py \
	--eval_only \
	--model_path ./out/dec23_oracle_reach2d_mlp_no_tanh/model_4.pt\
    --environment Reach2D \
    --method BC \
    --arch MLP \
	--hidden_size 20 \
    --num_models 1 \
    --seed 4 
