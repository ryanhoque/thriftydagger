#!/bin/bash

python run_thriftydagger.py \
        --gen_data \
        --environment PickPlace \
        --num_bc_episodes 1000 \
        --no_render \
        --algo_sup \
        dec14_gen_oracle_pick_place_data_1k