import torch 
import os 
from llava.conversation import conv_templates  
from torch.cuda.amp import autocast 
from torchvision import transforms 

from utils_attack import *

from setproctitle import setproctitle

from inversion_loss import *


def inversion_attack(args,model, tokenizer,  G, ws, input_ids,label_ids, attack_transform, file_path,device,hidden_states_mean, hidden_states_var,n_epochs=100, lr=0.05):
    
    num_ws = G.num_ws
    batchsize = args.batchsize
    input_ids = input_ids.expand(batchsize,input_ids.shape[1])
    label_ids = label_ids.expand(batchsize,label_ids.shape[1])
    

    if args.loss =="CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss =="NLLL" :
        criterion = torch.nn.NLLLoss()
    elif args.loss =="LOM":
        criterion = LOM    
    elif args.loss =="MML":
         criterion = max_margin_loss
         

    for i in range(int(ws.shape[0]/batchsize)):
        w_filepath = f"{file_path}{i}.pt"
        
        if os.path.exists(w_filepath) == False:                 
            w=torch.nn.Parameter(ws[i*batchsize:(i+1)*batchsize].detach(), requires_grad = True).to(device)
            
            optimizer = torch.optim.Adam([w], lr=lr, weight_decay=0, betas=[0.1, 0.1])
            all_loss = []
            grad =[]
            lrs =[]
            output_text = []
            all_w = []
            correct_class_probs = []
            minLoss= 10000
            finalStep = -1
            all_w.append(w.detach().clone().cpu())
    
            index = label_ids.shape[1]-sum(label_ids[0][:]==-100)+1
            
            label = label_ids[0][-index+1:-1]

            n_tokens = index - 2
            n_epoch_ = int(n_epochs/n_tokens)
            for token in range(n_tokens):
                for epoch in range(n_epoch_):               
                    image = synthesis(G,w,num_ws)
                    
                    image = clip_images(image)
                    image_tensor = attack_transform(image)
                    with autocast():
                        output =  model(
                            input_ids=input_ids,
                            images=image_tensor,
                            labels=label_ids  # using input_ids as target labels for causal LM loss
                        )
                        logits = output.logits[0]
                        output_ = logits[-index:-2]
                        
                        with torch.no_grad():                        
                            output_ids = torch.argmax(output_, dim=1)
                            probs = torch.nn.functional.softmax(output_, dim=1) 
                            correct_class_probs_ = probs.gather(1,label.unsqueeze(1))
                            correct_class_probs_ = correct_class_probs_.squeeze(1)
                            generated_text = tokenizer.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True)[0]
                            output_text.append(generated_text)
                        
                        output_ = output_[token].unsqueeze(0)
                        new_label = label[token].unsqueeze(0)
                        if args.loss =="LOM":                        
                            hidden_states = output.hidden_states[0][-index:-2]
                            hidden_states = hidden_states[token].unsqueeze(0)
                            loss = criterion(new_label, output_, hidden_states, hidden_states_mean, hidden_states_var) 
                        else:                  
                            loss = criterion(output_,new_label) 
                        



                        optimizer.zero_grad()
                        loss.backward()
                        all_loss.append(loss.item())
                        grad.append(w.grad.detach().clone().norm().item())
                        lrs.append(optimizer.param_groups[0]['lr'])
                        correct_class_probs.append(correct_class_probs_)
                        optimizer.step()
                        
                        
                        all_w.append(w.detach().clone().cpu())
                        
                        
                        if loss.item()<minLoss:
                            minLoss =loss.item()
                            finalW=w.detach().clone()
                            finalStep = epoch

                        print(f"Epoch {token}-{epoch} Loss = {loss.item()}, {generated_text}")
                    
            if args.save_img==True:
                image = synthesis(G,w,num_ws)
                if args.stylegan_model =="stylegan2-ada-pytorch/ffhq.pkl": #if ffhq
                    save_images(image, file_path,f'{i}_final_{generated_text}',800)
                else:
                    save_images(image, file_path,f'{i}_final_{generated_text}')
            correct_class_probs = torch.stack(correct_class_probs).detach().cpu()
            torch.save({'w':w.detach(), 'minW':finalW, 'output_text': output_text, 'all_w':all_w, 'finalStep': finalStep, 'correct_class_probs':correct_class_probs}, w_filepath)
            

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

    save_fol = f"inverted/{model_name}_{args.initW_path}_{args.savepath}/"
    os.makedirs(save_fol, exist_ok=True)
    with open(f"{save_fol}config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

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
            w = find_initial_w(G, model, 2000, input_id, label_id, attack_transform)
            torch.save({'w':w.detach()}, optimized_w_path_selected)
        else:
            print('loading-----')
            data_ = torch.load(optimized_w_path_selected)
            w = data_['w']
            del data_            
        save_fol = f"inverted/{model_name}_{args.initW_path}_{args.savepath}/{folder_name}/"
        print('save_fol',save_fol)
        os.makedirs(save_fol, exist_ok=True)
        
        
        if args.loss == "LOM":
            conv = conv_templates[conv_mode].copy()
            input_id_inference, _ =  create_token_inference(prompt_int,target_prompt,conv,tokenizer,device)
        
            hidden_states_mean, hidden_states_var=LOM_public_features(model,args.GAN_penultimate_activation, args.Dpub, input_id_inference, attack_transform, device)
        else:
            hidden_states_mean, hidden_states_var = None, None
        inversion_attack(args, model, tokenizer,  G, w, input_id,label_id,attack_transform,save_fol,device,hidden_states_mean, hidden_states_var, args.epoch, args.lr)

if __name__ == '__main__':
    main()
