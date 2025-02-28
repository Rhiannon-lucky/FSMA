import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torch_dct
from torchvision import transforms as T
from dct import *
import os
from PIL import Image
import kornia
import filter

"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

# """Input diversity: https://arxiv.org/abs/1803.06978"""
# def DI(x, resize_rate=1.15, diversity_prob=0.5):
#     assert resize_rate >= 1.0
#     assert diversity_prob >= 0.0 and diversity_prob <= 1.0
#     img_size = x.shape[-1]
#     img_resize = int(img_size * resize_rate)
#     rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
#     rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
#     h_rem = img_resize - rnd
#     w_rem = img_resize - rnd
#     pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
#     pad_bottom = h_rem - pad_top
#     pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
#     pad_right = w_rem - pad_left
#     padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
#     ret = padded if torch.rand(1) < diversity_prob else x
#     # print('++++++++++++', ret.shape)
#     return ret



def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    padded = F.interpolate(padded, size=img_size, mode='bilinear', align_corners=False)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def RP(x):
    img_size = x.shape[-1]
    img_resize = 331
    fs = torch.zeros((x.shape[0], 3, 331, 331))
    for i in range(x.shape[0]):
        rnd = torch.randint(low=310, high=img_resize, size=(1,), dtype=torch.int32)        
        image = F.interpolate(torch.unsqueeze(x[i], 0), size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
    
        padded = F.pad(image, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        fs[i] = padded
    
    imagess = F.interpolate(fs, size=img_size, mode='bilinear', align_corners=False)
    return imagess

def patch_transfrom(x, bound):
    pos = torch.randint(low=0, high=1000, size=(1,))
    t = 4
    if pos % t == 0:
        x = 0
    elif pos % t == 1:
        betas = torch.rand(size=(3,1,1))
        x = x * betas
    elif pos % t == 2:
        # scale
        x = x + torch.randn_like(x).uniform_(-bound, bound)
    else:
        sslist = [0,1,2,3,4]
        x = x/torch.pow(2, sslist[pos%5])
    
    return x

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def enhance_SF(x, im, n, N, ll, device, gamma):
    # patch_drop
    im = im.to(device)
    ll = (x.shape[2]+1)//2
    drop_mask = torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]))
    max_img = torch.ones_like(x)
    dct_x = dct_2d(x)
    dct_im = dct_2d(im)

    # max_img = dct_x
    # 1
    # max_img[:, :,0:ll,0:ll] = dct_x[:,:,0:ll,0:ll]
    max_img[:, :,0:ll,0:ll] = (1-n/N) * dct_x[:,:,0:ll,0:ll] + ((n)/N*(1/2**n))*dct_im[:,:,0:ll,0:ll]
    # # 2
    # max_img[:, :,ll:,0:ll] = dct_x[:,:,ll:,0:ll]
    max_img[:, :,ll:,0:ll] = (1-n/N)*dct_x[:,:,ll:,0:ll] + ((n)*(1/2**n))*dct_im[:,:,ll:,0:ll]
    # # 3
    # max_img[:, :,0:ll,ll:] = dct_x[:,:,0:ll,ll:]
    max_img[:, :,0:ll,ll:] = (1-n/N)*dct_x[:,:,0:ll,ll:] + ((n)*(1/2**n))*dct_im[:,:,0:ll,ll:]    
    # # 4
    # max_img[:, :,ll:,ll:] =  dct_x[:, :,ll:,ll:]
    max_img[:, :,ll:,ll:] =  (1-n/N) * dct_x[:, :,ll:,ll:] + ((n)/N)*dct_im[:,:,ll:,ll:]
    
    # dpt = torch.nn.Dropout2d(p=gamma)
    # max_img = dpt(max_img)
    max_img_sp = idct_2d(max_img)
    max_img_sp = torch.clip(max_img_sp, 0, 1)
    dpt = torch.nn.Dropout2d(p=gamma)
    max_img_sp = dpt(max_img_sp)
    return max_img_sp


# def save_image(images, output_dir):
#     """save the adversarial images"""
#     if os.path.exists(output_dir)==False:
#         os.makedirs(output_dir)
#     adv_img_np = np.transpose(images.cpu().numpy(), (0, 2, 3, 1)) * 255
#     r = torch.randint(10000,size=(images.shape[0],))
#     for i in range(images.shape[0]):
#         img = Image.fromarray(adv_img_np[i].astype('uint8'))  
#         img.save(os.path.join(output_dir, 'transfer'+str(r[i])+'.jpg'))