#!/bin/bash

python main.py \
		--data_path ./data/nov18-gen-data-pick-place/nov18-gen-data-pick-place_s4/pick-place-data-30.pkl \
		--environment PickPlace \
		--method HGDagger \
		--arch MLP \
		--num_models 5 \
		--trajectories_per_rollout 10 \
		--policy_train_epochs 5 \
		--epochs 5 \
		--robosuite