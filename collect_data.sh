### for GOOD demos ###
# python run_trained_agent.py \
#  --agent SquarePeg \
#  --eval_path datasets/square_machine_policy_more/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy_more/square_30_good.hdf5 \
#  --n_rollouts 30 --horizon 400 --seed 213

# python run_trained_agent.py \
#  --agent SquarePeg \
#  --eval_path datasets/square_machine_policy_more/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy_more/square_50_good.hdf5 \
#  --n_rollouts 50 --horizon 400 --seed 331

### for MIXED demos ###
# python run_trained_agent.py \
#  --agent SquarePeg \
#  --eval_skip 20 \
#  --eval_path datasets/square_machine_policy_more/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy_more/square_600_paired.hdf5 \
#  --n_rollouts 600 --horizon 400 --seed 123 --paired

# python run_trained_agent.py \
#  --agent SquarePeg \
#  --eval_skip 20 \
#  --eval_path datasets/square_machine_policy_more/ \
#  --reference_data datasets/square/ph/image.hdf5 \
#  --config configs/image_collection.json \
#  --machine_oracle --success_only --dataset_obs \
#  --dataset_path datasets/square_machine_policy_more/square_800_paired.hdf5 \
#  --n_rollouts 800 --horizon 400 --seed 456 --paired

#### HOW TO COLLECT EXPERT DEMOS FROM A PRETRAINED AGENT ####
#checkpoint=440
#python run_trained_agent.py \
#  --agent bc_trained_models/can_image/20220707164110/models/model_epoch_440.pth \
#  --checkpoint reset_$checkpoint --mode rollout  --dataset_path datasets/can/paired/image_10_good.hdf5\
#  --eval_path datasets/can/paired --dataset_obs\
#  --n_rollouts 10 --horizon 400 --success_only --seed 11
