#!/bin/bash

MUS=(0.0 1.0 5.0 50.0 100.0)
STDS=(0.1 0.5 1.0)

for mu in "${MUS[@]}"
do
	for std in "${STDS[@]}"
	do
		python test_uncertainty.py --exp_name noised_data_mu$mu\_std$std --noise_mu $mu --noise_std $std
	done
done
