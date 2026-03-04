# Model Inversion Attacks on Vision-Language Models: Do They Leak What They Learn?

---

# Environment

First install environments for LLaVA model
```
conda create -n llava python=3.10 -y 
conda activate llava
pip install -r requirements.txt


cd LLaVA/
pip install --upgrade pip # enable PEP 660 support
pip install -e .
pip install -e ".[train]" 


pip install flash-attn --no-build-isolation #optional
pip install flash-attn==2.3.6  #optional
```

# Clone Stylegan-ada
```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
rm -r --force stylegan2-ada-pytorch/.git/
rm -r --force stylegan2-ada-pytorch/.github/
rm --force stylegan2-ada-pytorch/.gitignore

wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P stylegan2-ada-pytorch/

```
# Training Model and evaluate
```
# Train model
bash scripts/train_llava_lora_clean.sh

```
## Natural accuracy of the model
```
python accuracy.py

```
# Attack

For TMI, TMI_C, and SMI, we provide 3 losses: `CE`, `LOM`, and `MML`.
For SMI_AW, we provide 3 losses: `CE`, ```LOM`, and `MML_batch`.

## Search for initial w (optional)


```
CUDA_VISIBLE_DEVICES=0 python MI_Sequence_based_SMI_AW.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--dataset=facescrub

```

## SMI_AW

```

CUDA_VISIBLE_DEVICES=0 python MI_Sequence_based_SMI_AW.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="SMI_AW" --loss=LOM \
--epoch=70 --Dpub="../../ffhq/" \
--GAN_penultimate_activation="metadata/ffhq_stylegan_facescrub.pt" \
--dataset=facescrub

```

## SMI

```

CUDA_VISIBLE_DEVICES=0 python MI_Sequence_based_SMI.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="SMI" --loss=LOM \
--epoch=70 \
--dataset=facescrub

``` 

## TMI

```

CUDA_VISIBLE_DEVICES=0 python MI_token_based_TMI.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="TMI" --loss=LOM \
--epoch=70 \
--dataset=facescrub

``` 

## TMI-C

```

CUDA_VISIBLE_DEVICES=0 python MI_token_based_TMI_C.py --save_img \
--istart=0 --iend=530 --model_name=llava-v1.6-vicuna-7b \
--savepath="TMI_C" --loss=LOM \
--epoch=70 \
--dataset=facescrub

``` 

### Evaluation:
```

CUDA_VISIBLE_DEVICES=0 python eval_MI.py --model_name=llava-v1.6-vicuna-7b \
--savepath=SMI_AW \
--initW_path=initW \
--iterations=10 --evalID=100 --loss=LOM \
--skip 0 530

``` 
## Implementation Credits

Some parts of our implementation are based on publicly available repositories. We sincerely thank the original authors for sharing their code.
- LLaVA: https://github.com/haotian-liu/LLaVA/
- PPA: https://github.com/LukasStruppek/Plug-and-Play-Attacks
- LOMMA: https://github.com/sutd-visual-computing-group/Re-thinking_MI/
- PLGMI: https://github.com/LetheSec/PLG-MI-Attack
- KEDMI: https://github.com/SCccc21/Knowledge-Enriched-DMI
- FID Score: https://github.com/mseitzer/pytorch-fid
- FaceNet: https://github.com/timesler/facenet-pytorch
