#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

dataset=$1

date
srun python train_DebGCD.py \
    --dataset_name ${dataset} \
    --batch_size 128 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --warmup_model_dir /home/ypliu0/projects/HypCD/pretrained/dino_vitb16_pretrain.pth \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --eval_only \
    --eval_path debgcd_models/dino${dino}/${dataset}/model.pt \
    --dino v1 > logs/eval_debgcd_dino${dino}_${dataset}.txt
date