#!/bin/bash

STDS=(0.1 0.5 1.0)


for std in "${STDS[@]}"
do
    python plot_uncertainty.py --std $std
done
