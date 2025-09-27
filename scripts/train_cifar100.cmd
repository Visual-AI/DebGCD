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

dino=v1
dataset=cifar100
mem=4.0
gfb=11
bsz=256
thr=0.55
adl=0.5
sdl=0.001
pl=0.5
we=30

date
srun python train_DebGCD.py \
    --dataset_name ${dataset} \
    --batch_size $bsz \
    --grad_from_block $gfb \
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
    --warmup_teacher_temp_epochs $we \
    --memax_weight $mem \
    --sdl_loss_weight $sdl \
    --adl_loss_weight $adl \
    --pl_loss_weight $pl \
    --threshold $thr \
    --dino $dino > logs/debgcd_dino${dino}_${dataset}_bsz${bsz}_gfb${gfb}_thr${thr}_adl${adl}_sdl${sdl}_pl${pl}_we${we}.txt
date