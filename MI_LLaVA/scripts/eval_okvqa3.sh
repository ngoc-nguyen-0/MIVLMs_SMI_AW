 

#  CUDA_VISIBLE_DEVICES=3 python model_vqa_our.py \
#     --model-path "./checkpoints/train_90/liuhaotian/llava-v1.5-7b/" \
#     --model_name "llava_v1.5_lora" \
#     --model-base "liuhaotian/llava-v1.5-7b" \
#     --image-folder "/home/users/ngocntb2/facescrub/" \
#     --question-file "facescrub_test.json" \
#     --answers-file "./results/llava-v1.5-7b_facescrub_90_test_results_our_transform.json"




 CUDA_VISIBLE_DEVICES=3 python model_vqa_our.py \
    --model_name "llava-v1.6-vicuna-13b" \
    --image-folder "/home/users/ngocntb2/facescrub/" \
    --question-file "facescrub_test.json" \
    --answers-file "./results/llava-v1.6-vicuna-13b_facescrub_90_test_results_our_transform_2.json"
