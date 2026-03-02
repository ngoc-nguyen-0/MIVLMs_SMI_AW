import torch
import pickle
import sys
import os
import torchvision
import torchvision.transforms.functional as F   
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX 
from llava.mm_utils import tokenizer_image_token 
from llava.mm_utils import tokenizer_image_token  
import csv

from llava.utils import disable_torch_init
def load_llava_model(model):
    
    disable_torch_init()
    """Loads the trained LLaVA model."""
    
    #Facescrub
    if model =="llava-v1.6-vicuna-7b":
        model_path = "./checkpoints/train_90/liuhaotian/llava-v1.6-vicuna-7b/"
        model_name = "llava_v1.6_lora"
        model_base = "liuhaotian/llava-v1.6-vicuna-7b"

    # StanfordDogs
    elif model =="llava-v1.6-vicuna-7b_StanfordDogs":
        model_path = "./checkpoints/llava-v1.6-vicuna-7b_StanfordDogs/liuhaotian/llava-v1.6-vicuna-7b/"
        model_name = "llava_v1.6_lora"
        model_base = "liuhaotian/llava-v1.6-vicuna-7b"

    # CelebA
    elif model =="llava-v1.6-vicuna-7b_celeba_random_name_336":
        model_path = "./checkpoints/llava-v1.6-vicuna-7b_celeba_336/liuhaotian/llava-v1.6-vicuna-7b/"
        model_name = "llava_v1.6_lora"
        model_base = "liuhaotian/llava-v1.6-vicuna-7b"

    #pretrained model
    elif model == "llava-v1.6-vicuna-7b_pretrained": 
        model_path = "liuhaotian/llava-v1.6-vicuna-7b"
        model_name = "liuhaotian/llava-v1.6-vicuna-7b"
        model_base = None

    # Load model with vision encoder
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,  # Use None unless you have a base model
        model_name=model_name
    )
    return model,  tokenizer, image_processor

def load_generator(filepath):
    """Load pre-trained generator using the running average of the weights ('ema').

    Args:
        filepath (str): Path to .pkl file

    Returns:
        torch.nn.Module: G_ema from pickle
    """
    with open(filepath, 'rb') as f:
        sys.path.insert(0, 'stylegan2-ada-pytorch')
        G = pickle.load(f)['G_ema'].cuda()
    return G


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename

class Config:
    pass


def create_parser():
    ####################################
    #        Attack Preparation        #
    ####################################
    import argparse

    parser = argparse.ArgumentParser(description="Simple argparse example")

    parser.add_argument("--savepath", type=str, default="")
    parser.add_argument("--istart", type=int, default=0)
    parser.add_argument("--iend", type=int, default=530)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--isTrain", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default="llava-v1.6-vicuna-7b")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--DA", type=int, default=1)
    parser.add_argument("--save_img",  action='store_true')
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--initW", type=str, default="train")
    parser.add_argument("--initW_path", type=str, default="initW")    
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--target_prompt", type=str, default="Donald Trump")
    parser.add_argument("--prompt_int", type=str, default="<image>\nWho is the person in the image?")
    parser.add_argument("--stylegan_model", type=str, default="stylegan2-ada-pytorch/ffhq.pkl")
    parser.add_argument("--GAN_penultimate_activation", type=str, default="./metadata/ffhq_stylegan_facescrub.pt")
    parser.add_argument("--Dpub", type=str, default="../../ffhq/ffhq256/")
    parser.add_argument("--dataset", type=str, default="facescrub")
        
        
    
    parser.add_argument("--inner_loop", type=int, default=5)
    
    ########### evaluation
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples_per_target", type=int, default=8)
    parser.add_argument("--n_imgs", type=int, default=16)    
    parser.add_argument("--skip", type=int, nargs='+', default=[-1, 530])
    
    parser.add_argument("--evalID", type=int, default=530)
    parser.add_argument("--debug",  action='store_true')
    parser.add_argument("--unfiltered",  action='store_true')
    parser.add_argument("--w", type=str, default="w")

    

    args = parser.parse_args()
    args.savepath = args.loss+"_"+args.savepath
    return args

def save_images(imgs: torch.tensor, folder, filename, center_crop=None):
    """Save StyleGAN output images in file(s).

    Args:
        imgs (torch.tensor): generated images in [-1, 1] range
        folder (str): output folder
        filename (str): name of the files
    """
    imgs = imgs.detach()
    if center_crop:
        imgs = F.center_crop(imgs, (center_crop, center_crop))
    imgs = (imgs * 0.5 + 128 / 255).clamp(0, 1)
    for i, img in enumerate(imgs):
        path = os.path.join(folder, f'{filename}_{i}.png')
        torchvision.utils.save_image(img, path)
def create_w(G, batch_size, device, truncation_psi=0.7, truncation_cutoff=18, c=None):
   
    z = torch.randn(batch_size, G.z_dim).to(device)
    w = G.mapping(z,
                c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff)
    return w

def get_intended_token_ids(input_ids, target_id,debug = False):
    padding = torch.full_like(input_ids, -100)
    padding_dim = padding.shape[1]
    for i in range(1,len(target_id)):
        padding[:,padding_dim-len(target_id)+i] = target_id[i]
    if debug:
        print("input_ids is:",input_ids)
        print("target_id is:",target_id)
        print("padding is:",padding)
    return padding    

def synthesis(G,w,num_ws):
    w_expanded = torch.repeat_interleave(w,
                                            repeats=num_ws,
                                            dim=1)
    imgs = G.synthesis(w_expanded,
                            noise_mode='const',
                            force_fp32=True)
    return imgs
def clip_images( imgs):
    lower_limit = torch.tensor(-1.0).float().to(imgs.device)
    upper_limit = torch.tensor(1.0).float().to(imgs.device)
    imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
    imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
    return imgs

def create_token(prompt_int,target_prompt,conv,tokenizer,device):
    
    conv.append_message(conv.roles[0], prompt_int)
    conv.append_message(conv.roles[1], target_prompt)
    prompt = conv.get_prompt()

    input_id = tokenizer_image_token(prompt, tokenizer,IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    output_temp = tokenizer_image_token(f"{target_prompt}</s>", tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt')

    label_id = get_intended_token_ids(input_id,output_temp) 
    input_id = input_id.to(device)
    label_id = label_id.to(device)
    return input_id, label_id

def create_token_inference(prompt_int,target_prompt,conv,tokenizer,device,batchsize=1):
    
        
    conv.append_message(conv.roles[0], prompt_int)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_id = tokenizer_image_token(prompt, tokenizer,IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    label_id = tokenizer_image_token(f"{target_prompt}</s>", tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt')
    label_id = label_id[1:].unsqueeze(0)

    input_id = input_id.to(device)
    label_id = label_id.to(device)
    
    input_id = input_id.expand(batchsize,input_id.shape[1])
    label_id = label_id[0].repeat(batchsize)
    
    return input_id, label_id