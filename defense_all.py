# -*- coding:utf-8 -*-

import torch
from attack_methods import *
# from defence.feature_squeezing2 import  FS
from defence.feature_s_Bit import FeatureSqueezing
from defence.feature_distillation import FD_jpeg_encode
from defence.transdefenses import *
from defence.nrp.purify import purify_test
# from typing import TYPE_CHECKING, Callable, List, Dict, Optional, Tuple, Union
# CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]  # pylint: disable=C0103
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter
import torchvision.transforms as transforms

resize1 = transforms.Resize((299,299))
resize2 = transforms.Resize((224,224))
# applies the normalization transformations
def apply_normalization_imagenet(imgs, config={'mean':[0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}):
    mean = config['mean']
    std = config['std']
    imgs_tensor = imgs.clone()
    if imgs.dim() == 3:
        for i in range(imgs_tensor.size(0)):
            imgs_tensor[i, :, :] = (
                imgs_tensor[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(imgs_tensor.size(1)):
            imgs_tensor[:, i, :, :] = (
                imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    
    return imgs_tensor
# FS = FeatureSqueezing(clip_values=(0, 1), bit_depth=5)
bits_squeezing = BitSqueezing(bit_depth=5)
jpeg_filter = JPEGFilter(75)

jpegdefense = torch.nn.Sequential(
    jpeg_filter
)

bsdefense = torch.nn.Sequential(
    bits_squeezing
)

def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))  
        img.save(os.path.join(output_dir, 'defence-'+name))

DEFENCE_METHOD = {'RP':RP,
                  'FS':bsdefense,
                  'FD':FD_jpeg_encode,
                  'jpeg':jpegdefense,
                  'quantize':get_defense,
                  'NRP':purify_test,
                  'NRP-ori':purify_test}

def defence(modelens, model, x, y, a_adv, device, images_ID, opt):
    
    if 'vit' in opt.model_name or 'beit' in opt.model_name:
        adv_img1 = resize1(a_adv)
    else:
        adv_img1 = a_adv
    defences_num = {}
    for dm in DEFENCE_METHOD:
        defences_num[dm] = 0
    
    for dm in DEFENCE_METHOD:
        x_trans = torch.ones_like(adv_img1)
        if dm in ['RP', 'FD']:
            x_trans = DEFENCE_METHOD[dm](adv_img1)
        elif dm in ['FS', 'jpeg']:
            x_trans = DEFENCE_METHOD[dm](adv_img1)
        elif 'NRP' in dm:
            x_trans = purify_test(adv_img1, device)
            x_trans_2 = x_trans.clone()
        else:
            x_trans = DEFENCE_METHOD[dm](dm, adv_img1)(adv_img1)
        x_trans = torch.tensor(x_trans).to(device)
        adv_img_np = x_trans.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        # save_image(adv_img_np, images_ID, os.path.join(opt.output_dir, opt.att_name, opt.model_name, dm))

        
        logit = modelens(x_trans)[0]
        defences_num[dm] = (torch.argmax(logit, axis=1) != torch.tensor(y).to(device)).detach().sum().cpu()
        
        if dm == 'NRP-ori':
            if 'vit' in opt.model_name or 'beit' in opt.model_name:
                adv_img2 = resize2(x_trans)
                logit = model(apply_normalization_imagenet(adv_img2))
                defences_num[dm] = ((torch.argmax(logit, axis=1)+1) != torch.tensor(y).to(device)).detach().sum().cpu()
            else:
                adv_img2 = x_trans
                logit = model(adv_img2)[0]
                defences_num[dm] = (torch.argmax(logit, axis=1) != torch.tensor(y).to(device)).detach().sum().cpu()
        
    return defences_num
