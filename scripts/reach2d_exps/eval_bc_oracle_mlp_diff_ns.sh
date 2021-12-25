#!/bin/bash
NS=(50 100 200 300 400 500 750 1000)

for N in "${NS[@]}"
do
		python src/main.py \
			--exp_name dec23/oracle_reach2d_mlp_eval_N$N \
			--N $N \
			--eval_only \
			--model_path ./out/dec23/oracle_reach2d_mlp_N$N/model_4.pt\
			--environment Reach2D \
			--method BC \
			--arch MLP \
			--hidden_size 20 \
			--num_models 1 \
			--seed 4 
done
