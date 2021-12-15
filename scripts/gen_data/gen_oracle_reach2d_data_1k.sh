#!/bin/bash

python run_thriftydagger.py \
        --gen_data \
        --environment Reach2D \
        --bc_only \
        --num_bc_episodes 1000 \
        --no_render \
        --algo_sup \
        dec14_gen_oracle_reach2d_data_1k
