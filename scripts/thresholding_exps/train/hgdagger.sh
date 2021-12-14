#!/bin/bash
THRESHOLDS=(0.2 0.5 1.0 2.0 5.0)

for threshold in "${THRESHOLDS[@]}"
do
    python run_thriftydagger.py \
        --input_file /iliad/u/madeline/thriftydagger/data/testpickplace/testpickplace_s4/pick-place-data.pkl \
        --environment PickPlace \
        --no_render \
        --hgdagger \
        --algo_sup \
        --hg_oracle_thresh $threshold \
        --iters 5 \
        --num_test_episodes 10 \
        nov27_hgdagger_pick_place_threshold$threshold
done
