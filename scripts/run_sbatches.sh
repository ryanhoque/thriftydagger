#!/bin/bash

for experiment in /iliad/u/madeline/thriftydagger/scripts/thresholding_exps/extra-thresholds-eval/* 
do
    echo $experiment
    chmod u+x $experiment
    sbatch --partition=iliad --time=7-00:00:00 --cpus-per-task=4 --gres=gpu:1 --mem=16G --nodes=1 --ntasks-per-node=1 --error=${experiment}_err.log --output=${experiment}_out.log $experiment
    sleep 1
done
# done
echo "Done"

