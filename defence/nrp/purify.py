'''
Purify adversarial images within l_inf <= 16/255
'''

import torch
import os, sys
sys.path.append('/media/smarket/NVme02/caohan/WBMLS/Our_transfer/defence/nrp')
import argparse
from networks import *
# from NRPutils import *
import tqdm

def purify_test(img, device, purifier='NRP_resG', dynamic=False):

    if purifier == 'NRP':
        netG = NRP(3,3,64,23)
        netG.load_state_dict(torch.load('/media/smarket/NVme02/caohan/WBMLS/Our_transfer/defence/nrp/NRP_moderl/NRP.pth'))
    if purifier == 'NRP_resG':
        netG = NRP_resG(3, 3, 64, 23)
        netG.load_state_dict(torch.load('/media/smarket/NVme02/caohan/WBMLS/Our_transfer/defence/nrp/NRP_moderl/NRP_resG.pth'))
    netG = netG.to(device)
    netG.eval()
    # for p in netG.parameters():
    #     p.requires_grad = False

    for p in netG.parameters():
        p.requires_grad = False
    if dynamic:
        eps = 16/255
        img_m = img + torch.randn_like(img) * 0.05
        #  Projection
        img_m = torch.min(torch.max(img_m, img - eps), img + eps)
        img_m = torch.clamp(img_m, 0.0, 1.0)
    else:
        img_m = img

    purified = netG(img_m).detach()
    purified = torch.clamp(purified, 0.0, 1.0)
    return purified