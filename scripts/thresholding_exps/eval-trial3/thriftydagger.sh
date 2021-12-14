#!/bin/bash

ALPHAS=(0.01 0.1 0.25 0.5 1.0)

for alpha in "${ALPHAS[@]}"
do
    exp_name=nov27_thriftydagger_pick_place_alpha$alpha
    python run_thriftydagger.py \
        --input_file /iliad/u/madeline/thriftydagger/data/testpickplace/testpickplace_s4/pick-place-data.pkl \
        --environment PickPlace \
        --eval /iliad/u/madeline/thriftydagger/data/$exp_name/$exp_name\_s4/pyt_save/model.pt \
        --no_render \
        --algo_sup \
        --iters 0 \
        --num_test_episodes 100 \
        --targetrate $alpha \
        dec1_trial3_thriftydagger_auto_only_eval_pick_place_alpha$alpha
done
