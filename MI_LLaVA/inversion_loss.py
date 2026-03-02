
import torch
import os
import random
from torchvision import transforms
from PIL import Image

from torch.cuda.amp import autocast

def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def max_margin_loss_batch(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real) + margin


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu
def LOM_public_features(model, savepath, datapath, input_id_inference, attack_transform, device,max_images=2000):

    """
    For each image in datapath, load it, preprocess, and pass to the model.
    """
    if os.path.exists(savepath) == False:            
        image_files = [f for f in os.listdir(datapath) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        results = []
        attack_transform = transforms.Compose([
            transforms.Resize((336,336), interpolation=transforms.InterpolationMode.BICUBIC,  antialias=True),  # Resize shortest edge to 336
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711]) ]) # Normalize
        random.shuffle(image_files)
        image_files = image_files[:max_images]
        
        all_hidden_states = []
        for img_name in image_files:
            img_path = os.path.join(datapath, img_name)
            print('img_path',img_path)
            image = Image.open(img_path)#.convert('RGB')
            images = attack_transform(image)
            images = images.unsqueeze(0).half().to(device)
            with torch.no_grad():
                with autocast():
                    outputs = model.generate(
                        inputs=input_id_inference,
                        images=images,
                        return_dict_in_generate=True,                    
                        output_hidden_states=True,   
                        output_scores=True,                        
                        num_return_sequences=1
                    )
                    hidden_states = outputs.hidden_states
                    for i in range(1,len(hidden_states),1):
                        all_hidden_states.append(hidden_states[i].squeeze(0).squeeze(0))
        all_hidden_states =  torch.stack(all_hidden_states)
        fea_mean = torch.mean(all_hidden_states,dim=0)
        fea_logvar = torch.std(all_hidden_states,dim=0)
        
        
        torch.save({'fea_mean':fea_mean, 'fea_logvar':fea_logvar},savepath)
    else:
        data = torch.load(savepath)
        fea_mean=data['fea_mean']
        fea_logvar=data['fea_logvar']
    return fea_mean,fea_logvar



def reg_loss(featureT,fea_mean, fea_logvar):
    
    fea_reg = reparameterize(fea_mean, fea_logvar)
    fea_reg = fea_mean.repeat(featureT.shape[0],1)
    loss_reg = torch.mean((featureT - fea_reg).pow(2))
    return loss_reg

def LOM(label, logits, hidden_states, hidden_states_mean, hidden_states_var, w=None, alpha=0.1):
    if w==None:
        criterion = torch.nn.NLLLoss()
        loss = criterion(logits,label)
    else:
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = sum(criterion(logits,label)*w)

    loss_reg = reg_loss(hidden_states, hidden_states_mean, hidden_states_var)
    return loss + loss_reg*alpha
