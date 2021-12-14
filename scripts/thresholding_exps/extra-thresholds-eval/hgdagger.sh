#!/bin/bash
THRESHOLDS=(0.01 0.1)

for threshold in "${THRESHOLDS[@]}"
do
    exp_name=dec1_retrain_hgdagger_pick_place_threshold$threshold
    python run_thriftydagger.py \
        --input_file /iliad/u/madeline/thriftydagger/data/testpickplace/testpickplace_s4/pick-place-data.pkl \
        --environment PickPlace \
        --eval /iliad/u/madeline/thriftydagger/data/$exp_name/$exp_name\_s4/pyt_save/model4.pt \
        --no_render \
        --hgdagger \
        --algo_sup \
        --hg_oracle_thresh $threshold \
        --iters 0 \
        --num_test_episodes 100 \
        nov28_hgdagger_auto_only_eval_pick_place_threshold$threshold
done
