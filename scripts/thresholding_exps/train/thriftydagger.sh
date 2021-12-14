#!/bin/bash

ALPHAS=(0.01 0.1 0.25 0.5 1.0)

for alpha in "${ALPHAS[@]}"
do
    python run_thriftydagger.py \
        --input_file /iliad/u/madeline/thriftydagger/data/testpickplace/testpickplace_s4/pick-place-data.pkl \
        --environment PickPlace \
        --no_render \
        --algo_sup \
        --iters 5 \
        --num_test_episodes 10 \
        --targetrate $alpha \
        nov27_thriftydagger_pick_place_alpha$alpha
done
