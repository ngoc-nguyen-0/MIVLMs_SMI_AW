

CUDA_VISIBLE_DEVICES=1 python MI_Sequence_based_SMI_AW.py --save_img \
--istart=51 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="SMI_AW" --loss=LOM \
--epoch=70 --Dpub="../../ffhq/" \
--GAN_penultimate_activation="metadata/ffhq_stylegan_facescrub.pt" \
--dataset=facescrub