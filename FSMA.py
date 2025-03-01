"""Implementation of sample attack."""
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI, gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
# from dct import *
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import csv
from attack_methods import *
from Normalize import Normalize, TfNormalize
from guided_filter import GuidedFilter2d, FastGuidedFilter2d
from defense_all import *
import timm
from timm.data import resolve_data_config
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

from torch import nn
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
# num_workers=0

torch.manual_seed(0)
list_nets = [
    'tf2torch_inception_v3',
    'tf2torch_inception_v4',
    'tf2torch_resnet_v2_50',
    'tf2torch_resnet_v2_101',
    'tf2torch_resnet_v2_152',
    'tf2torch_inc_res_v2',
    'vit_small_patch16_224',
    'beit_base_patch16_224',
    'tf2torch_adv_inception_v3',
    'tf2torch_ens3_adv_inc_v3',
    'tf2torch_ens4_adv_inc_v3',
    'tf2torch_ens_adv_inc_res_v2'
    ]
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='tf2torch_inc_res_v2', help='source model name.') # 
    # parser.add_argument('--input_csv', type=str, default='/media/smarket/NVme02/dataset/nip2017/images.csv', help='Input csv with images.')
    # parser.add_argument('--input_dir', type=str, default='/media/smarket/NVme02/dataset/nip2017/images', help='Input images.')
    # parser.add_argument('--model_dir', type=str, default='/media/smarket/NVme02/dataset/nip2017/torchModel', help='Model weight directory.')
    parser.add_argument('--input_csv', type=str, default='/media/smarket/Dataset/data/dataset/nip2017/images.csv', help='Input csv with images.')
    parser.add_argument('--input_dir', type=str, default='/media/smarket/Dataset/data/dataset/nip2017/images', help='Input images.')
    parser.add_argument('--model_dir', type=str, default='/media/smarket/Dataset/data/dataset/nip2017/torchModel', help='Model weight directory.')


    parser.add_argument('--output_dir', type=str, default='defense-outputs', help='Output directory with adversarial images.') #
    parser.add_argument("--batch_size", type=int, default=5, help="How many images process at one time.") #
    parser.add_argument("--N", type=int, default=5, help="The copy number ")
    parser.add_argument('--line', type=int, default=150 , help='length parameter')
    parser.add_argument('--att_name', type=str, default='FSMA-dpt', help='attack mathod.')
    parser.add_argument('--gamma', type=float, default=0.1 , help='drop out')

    parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
    parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
    parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
    parser.add_argument("--momentum", type=float, default=1, help="Momentum")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")

    parser.add_argument("--filter", default="None", type=str)
    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--feps", default=1e-4, type=float)
    parser.add_argument("--fast", default=False, action="store_true")
    return parser.parse_args()

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1: # 
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square) # 
    return nor_grad

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

def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))  
        img.save(os.path.join(output_dir, 'adv-'+name))

T_kernel = gkern(7, 3) # 3,1,7,7

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    elif net_name == 'vit_small_patch16_224':
        net = timm.create_model(net_name, pretrained=True)#, checkpoint_path=os.path.join(modelsRoot, MODLES[m]))
        # config = resolve_data_config({}, model=model)
    elif net_name == 'beit_base_patch16_224':
        net = timm.create_model(net_name, pretrained=True)#, checkpoint_path=os.path.join(modelsRoot, MODLES[m]))
    else:
        print('Wrong model name!')

    if net_name != 'vit_small_patch16_224' and net_name != 'beit_base_patch16_224':
        model = nn.Sequential(
                TfNormalize('tensorflow'),
                net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    else:
        model = net.eval().to(device)
    return model

def save_gradient(gradients, names, output_dir, add_type):
    """save the adversarial images"""
    
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    for i in range(gradients.shape[0]):
        ss = gradients[i]
        print(ss.shape)
        means = torch.mean(ss)
        sigmas = torch.std(ss)
        maxss = means + 3 * sigmas
        minss = means - 3 * sigmas
        ss = torch.clip(ss, minss, maxss)
        ss_nor = (ss-ss.min())/(ss.max()-ss.min())
        ss_nor = ss_nor.cpu().numpy()
        ss_nor = np.transpose(ss_nor, (1, 2, 0)) * 255
        
        img = Image.fromarray(ss_nor[i].astype('uint8'))
        # print(os.path.join(output_dir, 'gradient-'+add_type+names[i]))
        img.save(os.path.join(output_dir, 'gradient-'+add_type+names[i]))


### Details will be completed as soon as the paper is accepted ###
def FSMA(images, gt, model, min, max, X, opt, images_ID):
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0

    alpha = eps / num_iter
    x = images.clone()
    x_copy = images.clone()
    
    noise = torch.zeros_like(x)
    old_grad = torch.zeros_like(x)
    N = opt.N

    for i in range(num_iter):
        ne_allgrad = 0
        mix_dataloader = DataLoader(X, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=0)  #  

        for n in range(N):
            randnoise = torch.randn_like(x).uniform_(-opt.beta*eps, opt.beta*eps)
            randnoise = randnoise.to(device)
            x_n = (x + randnoise).to(device)
            
            im, im_ID,  gt_new = next(iter(mix_dataloader))
            if 'vit' in opt.model_name or 'beit' in opt.model_name:
                im = resize2(im)
            x_n = enhance_SF(x_n, im, n, N, opt.line, device, opt.gamma)
            x_neighbor = V(x_n, requires_grad = True)
            # DI-FGSM https://arxiv.org/abs/1803.06978
            # ne_output = model(DI(x_neighbor))

            # ne_output = model(x_neighbor)
            
            # ne_output = model(x_neighbor)
            if 'vit' in opt.model_name or 'beit' in opt.model_name:
                if 'DI' in opt.att_name:
                    ne_output = model(DI(apply_normalization_imagenet(x_neighbor)))
                else:
                    ne_output = model(apply_normalization_imagenet(x_neighbor))
                ne_loss = F.cross_entropy(ne_output, gt)
            else:
                if 'DI' in opt.att_name:
                    ne_output = model(DI(x_neighbor))
                else:
                    ne_output = model(x_neighbor)
                ne_loss = F.cross_entropy(ne_output[0], gt)
            
            
            ne_loss.backward()
            ss = x_neighbor.grad.data
            ne_allgrad += ss
        
        ne_allgrad = ne_allgrad / N
        save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'none')
        ss_ori = ne_allgrad
        GF = GuidedFilter2d(opt.radius, opt.feps)
        
        if opt.filter == 'GF':
            if "enhanceX" in opt.att_name:
                ne_allgrad = GF(ne_allgrad, x_copy)
                save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'GFx')
        
            else:
                ne_allgrad = GF(ne_allgrad, ne_allgrad)
                save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'GFg')
        
        elif opt.filter == 'guass3':
            blur = kornia.filters.GaussianBlur2d((3, 3), (3, 3))
            ne_allgrad = blur(ne_allgrad)
            save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'guass3')
        
        elif opt.filter == 'guass1':
            blur = kornia.filters.GaussianBlur2d((3, 3), (1, 1))
            ne_allgrad = blur(ne_allgrad)
            save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'guass1')
        
        elif opt.filter == 'median':
            blur = kornia.filters.MedianBlur((3, 3))
            ne_allgrad = blur(ne_allgrad)
            save_gradient(ne_allgrad, images_ID, os.path.join('gradient_image', opt.model_name), 'median')
        else:
            pass
        
        noise = ne_allgrad
        
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        if 'TI' in opt.att_name:
            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        ## MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        
        noise = momentum * old_grad + noise
        old_grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets


##################################################################

def main(i, tt):
    opt = get_parser()
    opt.model_name = list_nets[i]
    opt.att_name = tt[0]
    opt.filter = tt[1]
    # DEFENCE_METHOD = {'RP', 'FS', 'FD', 'jpeg', 'tvm', 'quantize', 'quilting', 'NRP'}
    DEFENCE_METHOD = ['RP', 'FS', 'FD', 'jpeg', 'quantize', 'NRP', 'NRP-ori']
    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]

    models = get_models(list_nets, opt.model_dir)
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)  #  
    input_num = len(X)
    correct_num = {}
    
    logits = {}
    for net in list_nets:
        correct_num[net] = 0
    defence_num = {}
    for dess in DEFENCE_METHOD:
        defence_num[dess] = 0

    for images, images_ID,  gt_cpu in tqdm(data_loader):

        ## gt = gt_cpu.cuda()
        ## images = images.cuda()
        # print(torch.max(images), torch.min(images))
        gt = gt_cpu.to(device)
        images = images.to(device)              
        
        if 'vit' in opt.model_name or 'beit' in opt.model_name:
            images = resize2(images)
            gt = gt-1
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = FSMA(images, gt, model, images_min, images_max, X, opt, images_ID)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        # save_image(adv_img_np, images_ID, os.path.join(opt.output_dir, opt.att_name, opt.model_name))

        # Prediction
        with torch.no_grad():
            if 'vit' in opt.model_name or 'beit' in opt.model_name:
                gt = gt + 1
            for net in list_nets:
                # print(adv_img.shape)
                if 'vit' in opt.model_name or 'beit' in opt.model_name:
                    if 'vit' in net or 'beit' in net:
                        logits[net] = models[net](apply_normalization_imagenet(adv_img))#[0]
                        # print(logits[net].shape)
                    else:
                        adv_img1 = resize1(adv_img)
                        logits[net] = models[net](adv_img1)[0]
                else:
                    if 'vit' in net or 'beit' in net:
                        adv_img1 = resize2(adv_img)
                        logits[net] = models[net](apply_normalization_imagenet(adv_img1))#[0]
                    else:
                        logits[net] = models[net](adv_img)[0]
                # print(net, logits[net].shape, torch.argmax(logits[net], axis=1), gt)
                if 'vit' in net or 'beit' in net:
                    correct_num[net] += ((torch.argmax(logits[net], axis=1)+1) != gt).detach().sum().cpu()
                else:
                    correct_num[net] += (torch.argmax(logits[net], axis=1) != gt).detach().sum().cpu()
        
        deff_num = defence(models['tf2torch_ens_adv_inc_res_v2'], model, images, gt, adv_img, device, images_ID, opt)
        for dess in DEFENCE_METHOD:
            defence_num[dess] += deff_num[dess]
    # Print attack success rate
    with open(opt.att_name+'-'+opt.model_name+'-N-'+str(opt.N)+'-beta-'+str(opt.beta)+'-line-'+str(opt.line)+'-'+str(opt.gamma)+'-filter-'+opt.filter+'-r-'+str(opt.radius)+'-feps-'+str(opt.feps)+str(opt.fast)+'.csv', 'w') as f:
        wf = csv.writer(f)
        for net in list_nets:
            print('{} attack success rate: {:.2%}'.format(net, correct_num[net]/input_num))
            wf.writerow([net, correct_num[net]/input_num])
        for dess in DEFENCE_METHOD:
            print('{} attack success rate: {:.2%}'.format(dess, defence_num[dess]/input_num))
            wf.writerow([dess, defence_num[dess]/input_num])

if __name__ == '__main__':
    # att_name = [['FSMA-trans', 'None'], ['FSMA-trans-median', 'median'], ['FSMA-trans-guass3', 'guass'], ['FSMA-trans-avgGF', 'GF'], ['FSMA-trans-avgGF-enhanceX', 'GF']]
    
    # att_name = [['FSMA-trans-guass1', 'guass']]



    att_name = [['FSMA-trans-new-224-avgGF-enhanceX', 'GF'],
                ['FSMA-trans-new-224-avgGF', 'GF'],
                ['FSMA-trans-new-224-guass3', 'guass3'],
                ['FSMA-trans-new-224-guass1', 'guass1'],
                ['FSMA-trans-new-224-median', 'median']]
    # att_name = [['FSMA-trans-new-224-guass1', 'guass'],
    #             ['FSMA-trans-new-224-DI', 'None'],
    #             ['FSMA-trans-new-224-DI-avgGF-enhanceX', 'GF'],
    #             ['FSMA-trans-new-224-TI', 'None'],
    #             ['FSMA-trans-new-224-TI-avgGF-enhanceX', 'GF']]
    
    # att_name = [['FSMA-trans-22', 'none']]
    
    for tt in att_name:
        for i in range(0, 4):
            main(i, tt)
