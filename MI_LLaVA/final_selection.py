import torch
from evaluation.utils.stylegan import create_image
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from llava.conversation import conv_templates
from utils_attack import create_token

from setproctitle import setproctitle
from torch.cuda.amp import autocast
import os
def target_confidence(model,image,input_ids, target_ids,imgTransform):

    image_tensor = imgTransform(image)
    with autocast():
        loss = model(
            input_ids=input_ids,
            images=image_tensor,
            labels=target_ids,  # using input_ids as target labels for causal LM loss
        ).loss
    return -loss.item()
def scores_by_transform(imgs,
                        target_model,
                        transforms,
                        input_id, 
                        label_id,
                        iterations=100):

    score = 0

    with torch.no_grad():
        for i in range(iterations):
            prediction_vector =  target_confidence(target_model,imgs,input_id, label_id,transforms) 
            score += prediction_vector
        score = score / iterations
    return score


def perform_final_selection(w,
                            generator,
                            config,
                            targets,
                            target_model,
                            samples_per_target,
                            approach,
                            iterations,
                            idx_to_class,
                            conv_mode,
                            prompt_int,
                            tokenizer,
                            device,
                            skip=-1,
                            save_fol="./evaluation_results/",
                            rtpt=None):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_w = []
    if approach.strip() == 'transforms':
        transforms = T.Compose([
            T.RandomResizedCrop(size=(336, 336),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),
            T.RandomHorizontalFlip(0.5)
        ])
    iSkip=0
    step = 0
    for target in range(skip[0],skip[1],1):
        step+=1
        setproctitle(f"Perform final selection of IDs from {skip[0]} to {skip[1]}: {target}/{skip[1]-skip[0]}")
    

        target_files = f'{save_fol}{idx_to_class[target]}.pt'
        mask = torch.where(targets == target, True, False).cpu()
        
            
        if mask.any():
            
            mask = torch.where(targets == target, True, False).cpu()
            targets_masked = targets[mask].cpu()
            w_masked = w[mask]
            if os.path.exists(target_files) == False: 
                candidates = create_image(w_masked,
                                            generator,
                                            crop_size=config.attack_center_crop,
                                            resize=config.attack_resize,
                                            device=device).cpu()
                scores = []
                dataset = TensorDataset(candidates, targets_masked)
                
                target_prompt = idx_to_class[target].replace("_"," ")

                print('label',target_prompt)
            
                conv = conv_templates[conv_mode].copy()
                
                input_id, label_id =  create_token(prompt_int,target_prompt,conv,tokenizer,device)
                    
                
                for imgs, _ in DataLoader(dataset, batch_size=1):
                    imgs = imgs.to(device)

                    scores.append(
                        scores_by_transform(imgs, target_model, transforms, input_id, label_id,
                                            iterations))
                scores = torch.from_numpy(np.array(scores))
                torch.save({scores},target_files)
                
                indices = torch.sort(scores, descending=True).indices
                selected_indices = indices[:samples_per_target]
                final_targets.append(targets_masked[selected_indices].cpu())
                final_w.append(w_masked[selected_indices].cpu())

            else:
                scores = torch.load(target_files)                
                scores = list(scores)[0]
                
                indices = torch.sort(scores, descending=True).indices
                selected_indices = indices[:samples_per_target]
                final_targets.append(targets_masked[selected_indices].cpu())
                final_w.append(w_masked[selected_indices].cpu())

        if rtpt:
            rtpt.step(
                subtitle=f'Sample Selection step {step} of {skip[1]-skip[0]}'
            )
        iSkip+=1
    final_targets = torch.cat(final_targets, dim=0)
    final_w = torch.cat(final_w, dim=0)
    return final_w, final_targets
