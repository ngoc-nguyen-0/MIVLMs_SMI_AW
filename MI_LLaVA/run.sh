


CUDA_VISIBLE_DEVICES=0 python MI_token_based_TMI.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="TMI" --loss=LOM \
--epoch=70 \
--dataset=facescrub
