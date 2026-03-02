import torch
import os
from llava.conversation import conv_templates
from torch.cuda.amp import autocast
from torchvision import transforms
from utils_attack import *
import numpy as np
from setproctitle import setproctitle

           
def main():
    ####################################
    #        Attack Preparation        #
    ####################################
    
    args = create_parser()
    prompt_int = args.prompt_int

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stylegan_model= args.stylegan_model
    G = load_generator(stylegan_model)
    G.to(device)

    model_name = args.model_name
    model, tokenizer, _ = load_llava_model(model_name)
    model.to(device)
    if args.isTrain==-1:
        model.eval()
    else:
        model.config.gradient_checkpointing = True  # Set on config
        model.gradient_checkpointing_enable()       # Actually activate it
        model.enable_input_require_grads()          # Optional: for full grad support
        model.train()     

    conv_mode = "llava_v1"
    from initial_selection import find_initial_w

    attack_transform = transforms.Compose([      
            transforms.Resize((336,336), interpolation=transforms.InterpolationMode.BICUBIC,  antialias=True),  # Resize shortest edge to 336
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711]) ]) # Normalize
    import json  

    index = 0
    import json  
    if args.dataset =="facescrub":
        with open('metadata/facescrub_idx_to_class.pkl', 'rb') as f:
            idx_to_class = pickle.load(f)
        idx_to_class[93] = 'Freddy_Prinze_Jr_'
        idx_to_class[107] = 'Harry_Connick_Jr_'
        idx_to_class[238] = 'Robert_Downey_Jr_'
    elif args.dataset == "celeba":
        with open('metadata/celeba_idx_to_class.pkl', 'rb') as f:
            idx_to_class = pickle.load(f)
    elif args.dataset == "StanfordDogs":
        with open('metadata/stanford_dogs_idx_to_class.pkl', 'rb') as f:
            idx_to_class = pickle.load(f)
        prompt_int = "<image>\nWhat breed is this dog?"
    
   

    for index in range(args.istart, args.iend):
            
        conv = conv_templates[conv_mode].copy()
        target_prompt = idx_to_class[index].replace("_", " ")
            
        setproctitle(f"Invert IDs from {args.istart} to {args.iend}: {index}/{args.iend - args.istart}")
        input_id, label_id =  create_token(prompt_int,target_prompt,conv,tokenizer,device)
        
        folder_name = target_prompt.replace(" ","_")
        
        
        save_fol = f"inverted/{model_name}_{args.initW_path}/{folder_name}/"
        os.makedirs(save_fol, exist_ok=True)
        optimized_w_path_selected = f"{save_fol}init_w.pt"

        if os.path.exists(optimized_w_path_selected) == False: 
            w = find_initial_w(G, model, 2000, input_id, label_id, attack_transform,batch_size=25)
            torch.save({'w':w.detach()}, optimized_w_path_selected)
        
if __name__ == '__main__':
    main()
