#!/bin/bash
TAU_SUPS=(0.032 0.016 0.008 0.004 0.002)
TAU_AUTOS=(0.01 0.1 0.25 0.5 1.0)

for tau_sup in "${TAU_SUPS[@]}"
do
    for tau_auto in "${TAU_AUTOS[@]}"
    do
        exp_name=nov27_lazydagger_pick_place_tau_sup$tau_sup\_tau_auto$tau_auto
        python run_thriftydagger.py \
            --input_file /iliad/u/madeline/thriftydagger/data/testpickplace/testpickplace_s4/pick-place-data.pkl \
            --environment PickPlace \
            --eval /iliad/u/madeline/thriftydagger/data/$exp_name/$exp_name\_s4/pyt_save/model.pt \
            --no_render \
            --lazydagger \
            --algo_sup \
            --iters 0 \
            --num_test_episodes 100 \
            --tau_sup $tau_sup \
            --tau_auto $tau_auto \
            nov28_lazydagger_auto_only_eval_pick_place_tau_sup$tau_sup\_tau_auto$tau_auto
    done
done
