#!/bin/bash

# for ((i = 1; i <= 20; i++))
# do
#     python mtl/cgc_eval_by_epoch.py "$((5 * i))"
# done
# python mtl/train_fdn2.py

for ((i = 1; i <= 10; i++))
do
    python mtl/fdn2_eval_dup.py "$((1 * i))"
done

