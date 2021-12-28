#!/bin/bash
python src/main.py \
	--exp_name dec25/ensemble_dagger_reach2d_mlp_eval \
	--eval_only \
	--model_path ./out/dec25/ensemble_dagger_reach2d_mlp/model_4.pt\
    --environment Reach2D \
    --method Dagger \
    --arch MLP \
	--hidden_size 20 \
    --num_models 5 \
    --seed 4 
