import torch 
import os
from utils_attack import *

from evaluation.utils.stylegan import create_image

datasets = ['facescrub','celeba','StanfordDogs']
index = 0


dataset = datasets[index]
inverted_path = "./inverted/llava-v1.6-vicuna-7b_initW_CE_LOM_TSU_2/"
output_path = f"./inverted_images/{dataset}/TSU_LOM_2/"

if dataset=='facescrub':
    N=530
    stylegan_model ="stylegan2-ada-pytorch/ffhq.pkl"
    
elif dataset=='celeba':
    N=1000
    stylegan_model ="stylegan2-ada-pytorch/ffhq.pkl"
elif dataset=='stanford_dogs':
    N=120
    stylegan_model ="stylegan2-ada-pytorch/afhqdog.pkl"

config = Config()
config.attack_resize = 336
config.seed = 42 


if stylegan_model =="stylegan2-ada-pytorch/ffhq.pkl":
    config.attack_center_crop = 800
else:
    config.attack_center_crop = None
############################33

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = load_generator(stylegan_model)
G.to(device)
synthesis = G.synthesis

data = torch.load(f'{inverted_path}optimized_w_selected_w.pt')
w = data['w']
label = data['targets']

for i in range(N):
    mask = torch.where(label == i, True, False).cpu()
    targets_masked = label[mask].cpu()
    w_masked = w[mask].to(device)
    if w_masked.shape[0] <8:
        print(i)
    else:
        save_fol = f'{output_path}{i}/'
        os.makedirs(save_fol, exist_ok=True)
        img = create_image(w_masked,
                        synthesis,
                        crop_size=config.attack_center_crop,
                        resize=config.attack_resize,
                        device=device).cpu()
        save_images(img,save_fol,f'{i}')
