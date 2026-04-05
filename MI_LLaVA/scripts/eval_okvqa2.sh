 

#  CUDA_VISIBLE_DEVICES=0,1 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.6-vicuna-7b/" \
#     --model_name "llava_v1.6_lora" \
#     --model-base "liuhaotian/llava-v1.6-vicuna-7b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.6-vicuna-7b_facescrub_90_test_results.json"



 CUDA_VISIBLE_DEVICES=2 python model_vqa_our.py \
    --model_name "llava-v1.6-vicuna-13b" \
    --image-folder "/home/users/ngocntb2/facescrub/" \
    --question-file "facescrub_train.json" \
    --answers-file "./results/llava-v1.6-vicuna-13b_facescrub_90_train_results_our_transform_2.json"



# CUDA_VISIBLE_DEVICES=2 python model_vqa_our.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-13b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-13b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_train.json" \
#     --answers-file "./results/Inversion.json"


# CUDA_VISIBLE_DEVICES=2 python model_vqa_our.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-13b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-13b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_train.json" \
#     --answers-file "./results/llava-v1.5-13b_facescrub_90_train_results_our_transform.json"

# CUDA_VISIBLE_DEVICES=2 python model_vqa_our.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-13b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-13b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.5-13b_facescrub_90_test_results_our_transform.json"

#  CUDA_VISIBLE_DEVICES=4,5 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-13b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-13b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.5-13b_facescrub_90_test_results.json"

#  CUDA_VISIBLE_DEVICES=4,5 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-7b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-7b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.5-7b_facescrub_90_test_results.json"


# CUDA_VISIBLE_DEVICES=1,2,3,4 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-7b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-7b" \
#     --image-folder "/local_data/ngocntb/facescrub/" \
#     --question-file "facescrub_train.json" \
#     --answers-file "./results/llava-v1.5-7b_facescrub_90_train_results.json"

#  CUDA_VISIBLE_DEVICES=1,2,3,4 python LLaVA/llava/eval/model_vqa_science.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.6-vicuna-7b/" \
#     --model_name "llava_v1.6_lora" \
#     --model-base "liuhaotian/llava-v1.6-vicuna-7b" \
#     --image-folder "/local_data/ngocntb/facescrub/" \
#     --question-file "facescrub_train.json" \
#     --answers-file "./results/llava-v1.6-vicuna-7b_facescrub_90_train_results.json"

