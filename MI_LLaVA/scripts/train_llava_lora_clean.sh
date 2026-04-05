#!/bin/bash

##### argument to modify
# for llava_augTrain_lavisCLIP (training llava using data augmentation), need to modify LLaVA/llava/model/multimodal_encoder/clip_encoder.py; don't run other llava related experiments in the mean time.
GPU_ID=0,1
task_name=facescrub # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target
model_setting=llava # choose from: llava, instructBLIP_to_llava, miniGPT4v2_to_llava, llava_jpeg, llava_aug_lavisCLIP, llava_augTrain_lavisCLIP, llava_aug_lavisCLIP_jpeg, llava_augTrainLavisCLIP_noAugPoison, llava_aug_jpeg_jpeg
defence=clean-images
seed=0

SAVE_ROOT=. # change to your root for saving the poisoned VLMs

##### the following are automatic
clean_data_name=cc_sbu_align
master_port=$(( $GPU_ID+ 29000 ))
# if 2 GPU, acc_step=4; if 1 GPU, acc_step=8; effective bs = per_device_bs * accumulation * num_GPU should be 128 
gradient_accumulation_steps=4
per_device_train_batch_size=16
num_train_epochs=5


#############3 llava-v1.5-7b
# data_path=celeba_336x336_random_name_train.json
# data_root=/home/ngocntb/celeba_336x336/img_align_celeba/
# model=liuhaotian/llava-v1.6-vicuna-7b
# output_dir=$SAVE_ROOT/checkpoints/llava-v1.6-vicuna-7b_celeba_336_testttt/$model/
# deepspeed="LLaVA/scripts/zero3.json"

# deepspeed --include localhost:$GPU_ID --master_port $master_port LLaVA/llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed $deepspeed \
#     --model_name_or_path $model \
#     --version v1 \
#     --data_path $data_path \
#     --image_folder $data_root \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $output_dir \
#     --num_train_epochs $num_train_epochs \
#     --per_device_train_batch_size $per_device_train_batch_size \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb



#  CUDA_VISIBLE_DEVICES=0 python model_vqa_our.py \
#     --model_name "llava-v1.6-vicuna-7b_celeba_random_name_336" \
#     --image-folder "/home/ngocntb/celeba_336x336/img_align_celeba/" \
#     --question-file "celeba_336x336_random_name_test.json" \
#     --answers-file "./results/llava-v1.6-vicuna-7b_celeba_random_name_336_test.json"



# #############3 liuhaotian/llava-v1.5-13b
data_path=facescrub_train.json
data_root=/home/users/ngocntb2/facescrub/
model=liuhaotian/llava-v1.6-vicuna-7b
output_dir=$SAVE_ROOT/checkpoints/train_90_new/$model/
deepspeed="LLaVA/scripts/zero3.json"

deepspeed --include localhost:$GPU_ID --master_port $master_port LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed $deepspeed \
    --model_name_or_path $model \
    --version v1 \
    --data_path $data_path \
    --image_folder $data_root \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
######################################
#  CUDA_VISIBLE_DEVICES=1,2,3,4 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-7b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-7b" \
#     --image-folder "/local_data/ngocntb/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.5-7b_facescrub_90_test_results.json"

# data_path=facescrub_train.json
# data_root=/local_data/ngocntb/facescrub/
# model=liuhaotian/llava-v1.6-vicuna-7b
# output_dir=$SAVE_ROOT/checkpoints/MIDRE/$model/
# deepspeed="LLaVA/scripts/zero3.json"

# deepspeed --include localhost:$GPU_ID --master_port $master_port LLaVA/llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed $deepspeed \
#     --model_name_or_path $model \
#     --version v1 \
#     --data_path $data_path \
#     --image_folder $data_root \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $output_dir \
#     --num_train_epochs $num_train_epochs \
#     --per_device_train_batch_size $per_device_train_batch_size \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

