#!/bin/bash
### Arguments ###
# use --threshold = 0 for non-weighted (equivalent to balanced batches)
# use --good_only for expert demos only
# use --pretrained_agent to load a pretrained model
# use --dataset to specify the large D_prior dataset
# use --sample_data to specify the small expert D_t dataset
# use --classifier to specify the embedder used for BehaviorRetrieval

### Baselines ###
## no selection baseline ##
# seed=3
# base_dir=square_paired_lowdim_selected_BC
# run_name=no_selection_baseline_600_$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_600_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_lowdim_k0001/20230508122641/models/model_epoch_2000.pth \
#     sample_data=datasets/square_machine_policy_more/square_50_good.hdf5 train_size=50 threshold=0 seed=$seed \
#     eval_path=bc_trained_models/$base_dir/$run_name/ 

## behavior retrieval ##
seed=2
base_dir=square_paired_image_selected_BC
run_name=threshold_085_VAE_$seed

python simple_sbatch.py \
  --time 120:00:00 \
  --nodes 1 \
  --cpus 4 \
  --gpus titanxp:1 \
  --mem 32G \
  --job-name $run_name \
  --entry-point run_weighted_BC.py \
  --arguments \
    config=configs/bc_rnn_image.json name=$run_name \
    dataset=datasets/square_machine_policy_more/square_400_paired.hdf5 \
    classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
    sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0.85 seed=$seed \
    eval_path=bc_trained_models/$base_dir/$run_name/ 

## expert only ##
#seed=1
#base_dir=square_paired_image_selected_BC
#run_name=expert_only_10_$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_image.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_400_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
#     sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     eval_path=bc_trained_models/$base_dir/$run_name/ good_only

## oracle selection ##
#seed=1
#base_dir=square_paired_image_selected_BC
#run_name=oracle_selection_10_$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_image.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_400_paired_ORACLE.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
#     sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     eval_path=bc_trained_models/$base_dir/$run_name/ 

## goal conditioned (GCBC and Play-LMP) ##
# seed=2
# base_dir=square_paired_image_selected_BC
# run_name=goal_conditioned_baseline_$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_image_goal.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_400_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
#     sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     eval_path=bc_trained_models/$base_dir/$run_name/ goal

# seed=2
# base_dir=square_paired_lowdim_selected_BC
# run_name=LMP_baseline_600_$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_goal_lmp.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_600_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_lowdim_k0001/20230508122641/models/model_epoch_2000.pth \
#     sample_data=datasets/square_machine_policy/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     eval_path=bc_trained_models/$base_dir/$run_name/ goal

## fine tuned ##
#seed=200
#base_dir=square_paired_image_selected_BC
#run_name=finetuned$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_image.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_400_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
#     sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     pretrained_agent=bc_trained_models/square_paired_machine_image/20220908111304/models/model_epoch_400.pth \
#     eval_path=bc_trained_models/$base_dir/$run_name/

## goal conditioned with fine tuning ##
# seed=102
# base_dir=square_paired_image_selected_BC
# run_name=finetuned_goal_conditioned$seed

# python simple_sbatch.py \
#   --time 120:00:00 \
#   --nodes 1 \
#   --cpus 4 \
#   --gpus titanxp:1 \
#   --mem 32G \
#   --job-name $run_name \
#   --entry-point run_weighted_BC.py \
#   --arguments \
#     config=configs/bc_rnn_image_goal.json name=$run_name \
#     dataset=datasets/square_machine_policy_more/square_400_paired.hdf5 \
#     classifier=bc_trained_models/Embedder_VAE_square_image_k0001/20230508122641/models/model_epoch_1000.pth \
#     sample_data=datasets/square_machine_policy_more/square_10_good.hdf5 train_size=10 threshold=0 seed=$seed \
#     pretrained_agent=bc_trained_models/square_paired_image_selected_BC/fixed_goal_conditioned_baseline1/model_epoch_600.pth \
#     eval_path=bc_trained_models/$base_dir/$run_name/