#!/bin/bash
NS=(50 100 200 300 400 500 750 1000)

for N in "${NS[@]}"
do
		python main.py \
			--exp_name dec23/oracle_reach2d_mlp_N$N \
			--N $N \
			--data_path ./data/scripted_oracle_reach2d.pkl \
			--environment Reach2D \
			--method BC \
			--arch MLP \
			--hidden_size 20 \
			--num_models 1 \
			--seed 4 
done
