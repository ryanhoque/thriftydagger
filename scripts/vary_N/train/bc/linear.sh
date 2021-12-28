#!/bin/bash
ARCH=LinearModel
DATA_SOURCES=(oracle pi_r oracle_pi_r_mix)
DATE=dec28
ENVIRONMENT=Reach2D
METHOD=BC
NS=(50 100 200 300 400 500 750 1000)
NUM_MODELS=1
SEED=4

if [ $NUM_MODELS -gt 1 ]
then
    EXP_NAME_ARCH=Ensemble$ARCH
else
    EXP_NAME_ARCH=$ARCH
fi

for N in "${NS[@]}"
do
    for DATA_SOURCE in "${DATA_SOURCES[@]}"
    do
        python src/main.py \
            --N $N \
            --exp_name $DATE/$ENVIRONMENT/$METHOD/$EXP_NAME_ARCH/$DATA_SOURCE\_N$N\_seed$SEED \
            --data_path ./data/$ENVIRONMENT/$DATA_SOURCE.pkl \
            --environment $ENVIRONMENT \
            --method $METHOD \
            --arch $ARCH \
            --num_models $NUM_MODELS \
            --seed $SEED
    done
done
