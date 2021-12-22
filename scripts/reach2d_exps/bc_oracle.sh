#!/bin/bash
python main.py \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch LinearModel \
    --num_models 1 \
    --seed 4
