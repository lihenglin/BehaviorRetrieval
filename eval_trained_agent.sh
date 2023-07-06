#!/bin/bash

## BC agent ##
checkpoint=100
python run_trained_agent.py \
 --agent bc_trained_models/square_paired_lowdim_corrections/play_lmp_full_tighter_new_play_dagger_from_good_only_50_2/model_corrections_100.pth \
 --config configs/bc_rnn.json \
 --checkpoint $checkpoint \
 --n_rollouts 100 --horizon 400 --seed 1

## goal-conditioned BC agent ##
# checkpoint=10000
# python run_trained_agent.py \
#  --agent bc_trained_models/square_paired_lowdim_selected_BC/LMP_baseline_600_2/model_epoch_10000.pth \
#  --checkpoint $checkpoint  \
#  --n_rollouts 100 --horizon 400 --seed 1 --goal_data datasets/square_machine_policy_more/square_50_good.hdf5 