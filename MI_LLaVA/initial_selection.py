import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
import gc
from torch.cuda.amp import autocast
def target_confidence(model,image,input_ids, target_ids,imgTransform,initW):

    image_tensor = imgTransform(image)
    
    with torch.no_grad():
        with autocast():
            loss = model(
                input_ids=input_ids,
                images=image_tensor,
                labels=target_ids,  # using input_ids as target labels for causal LM loss
            ).loss
    
    return -loss.item()


def find_initial_w(generator,
                   target_model,                   
                   search_space_size,
                   input_ids, 
                   target_ids,
                   imgTransform,
                   initW ='train',
                   topn=16,
                   clip=True,
                   center_crop=768,
                   resize=336,
                   horizontal_flip=True,
                   filepath=None,
                   truncation_psi=0.7,
                   truncation_cutoff=18,
                   batch_size=25,
                   seed=0):
    """Find good initial starting points in the style space.

    Args:
        generator (torch.nn.Module): StyleGAN2 model
        target_model (torch.nn.Module): [description]
        target_cls (int): index of target class.
        search_space_size (int): number of potential style vectors.
        clip (boolean, optional): clip images to [-1, 1]. Defaults to True.
        center_crop (int, optional): size of the center crop. Defaults 768.
        resize (int, optional): size for the resizing operation. Defaults to 224.
        horizontal_flip (boolean, optional): apply horizontal flipping to images. Defaults to true.
        filepath (str): filepath to save candidates.
        truncation_psi (float, optional): truncation factor. Defaults to 0.7.
        truncation_cutoff (int, optional): truncation cutoff. Defaults to 18.
        batch_size (int, optional): batch size. Defaults to 25.

    Returns:
        torch.tensor: style vectors with highest confidences on the target model and target class.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.from_numpy(
        np.random.RandomState(seed).randn(search_space_size,
                                          generator.z_dim)).to(device)
    
    c = None
    five_crop = None

    with torch.no_grad():
        confidences = []
        final_candidates = []
        final_confidences = []
        candidates = generator.mapping(z,
                                       c,
                                       truncation_psi=truncation_psi,
                                       truncation_cutoff=truncation_cutoff)
        candidate_dataset = torch.utils.data.TensorDataset(candidates)
        for w in tqdm(torch.utils.data.DataLoader(candidate_dataset,
                                                  batch_size=batch_size),
                      desc='Find initial style vector w'):
            w = w[0]
            w = w[:,0,:].unsqueeze(1)
            w = torch.repeat_interleave(w,
                                        repeats=generator.num_ws,
                                        dim=1)
            
            imgs = generator.synthesis(w,
                                       noise_mode='const',
                                       force_fp32=True)
            if clip:
                lower_bound = torch.tensor(-1.0).float().to(imgs.device)
                upper_bound = torch.tensor(1.0).float().to(imgs.device)
                imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
                imgs = torch.where(imgs < lower_bound, lower_bound, imgs)
            if horizontal_flip:
                imgs_hflip = F.hflip(imgs)

            target_conf = None
            for idx in range(batch_size):
                if horizontal_flip:
                    im = torch.cat((imgs[idx].unsqueeze(0),imgs_hflip[idx].unsqueeze(0)),dim=0)
                    
                else:
                    im = imgs[idx].unsqueeze(0)
                target_conf =  target_confidence(target_model,im,input_ids, target_ids, imgTransform,initW) 

                confidences.append(target_conf)
        sorted_idx = [i for i, _ in sorted(enumerate(confidences), key=lambda x: x[1], reverse=True)]
       
        final_candidates = candidates[sorted_idx[:topn]]
       
        final_candidates = final_candidates[:,0,:].unsqueeze(1)

    if filepath:
        torch.save(final_candidates, filepath)
        print(f'Candidates have been saved to {filepath}')
    
    gc.collect()
    torch.cuda.empty_cache()
    return final_candidates
