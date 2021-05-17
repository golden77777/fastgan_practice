import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision import transforms

import os
import random
import argparse
from tqdm import tqdm

from models import GeneratorConditional

from operation import InfiniteWeightedSamplerWrapper


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)
    parser.add_argument('--src', type=str, default='images')
    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--condition', type=int, default=0)
    parser.set_defaults(big=False)
    args = parser.parse_args()
    batch_size = 1
    nlr = 0.0002
    nbeta1 = 0.5
    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))

    net_ig = GeneratorConditional( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)
    dataloader_workers = 8
    transform_list = [
            transforms.Resize((int(args.im_size),int(args.im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    dataset = ImageFolder(root='images2', transform=trans)
    #dataloader = iter(DataLoader(dataset, batch_size=batch_size shuffle=False, sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    mse = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        #ckpt = './models/%d.pth'%(epoch)
        checkpoint = torch.load(args.ckpt, map_location=lambda a,b: a)
        net_ig.load_state_dict(checkpoint['g'])

        optimizer = optim.Adam(net_ig.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        net_ig.eval()
        for i, (target_image, (path, category)) in enumerate(zip(dataloader,dataset.samples)):
            # noiseごとにcondition0,1の画像を作成
            noise = torch.randn(args.batch, noise_dim).to(device).requires_grad_(True)

            for condition in [0,1]:
                if condition == 0:
                    condition_code = F.one_hot(torch.zeros((args.batch,),dtype=torch.int64),num_classes=2).to(device)
                else :
                    condition_code = F.one_hot(torch.ones((args.batch,),dtype=torch.int64),num_classes=2).to(device)

                dist = 'eval_condition=%d_%d'%(condition,epoch)
                dist = os.path.join(dist, 'img')
                os.makedirs(dist, exist_ok=True)
                for i in range(10):
                  optimizer.zero_grad()
                  g_imgs = net_ig(noise,condition_code)[0]
                  loss=mse(target_image, g_imgs)
                  loss.backward()
                  optimizer.step()
                g_imgs = net_ig(noise,condition_code)[0]
                g_imgs = F.interpolate(g_imgs, 512)

                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5),
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
