#!/bin/bash

device='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES=$device

# export PATH=$PATH:path/to/robot-flamingo/robot_flamingo
# export PYTHONPATH=$PYTHONPATH:path/to/robot-flamingo/robot_flamingo
export PATH=$PATH:/share/dmh/cobra/cobra/cobra/cobra
export PYTHONPATH=$PYTHONPATH:/share/dmh/cobra/cobra/cobra/cobra
export PATH=$PATH:/share/dmh/cobra/cobra/cobra/cobra/robot_flamingo
export PYTHONPATH=$PYTHONPATH:/share/dmh/cobra/cobra/cobra/cobra/robot_flamingo
export PATH=$PATH:/share/dmh/hydra/open_flamingo
export PYTHONPATH=$PYTHONPATH:/share/dmh/hydra/open_flamingo

# dataset path
calvin_dataset_path='/share/dmh/cobra/cobra/cobra/dataset/task_ABCD_D'
# language model path
lm_path="/share/dmh/cobra/cobra/cobra/dataset/mamba-790m-hf"
# tokenizer path
tokenizer_path="/share/dmh/cobra/cobra/cobra/dataset/mamba-790m-hf"
# openflamingo ckpt path
openflamingo_checkpoint="/share/dmh/cobra/cobra/cobra/dataset/mamba-790m-hf/model.safetensors"

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"
# source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mamba_790m_hf \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --openflamingo_checkpoint ${openflamingo_checkpoint} \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --from_scratch \
    --window_size 12 
    # > ${log_file} 2>&1
