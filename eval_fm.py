import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator, Discriminator


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


noise_dim = 256
device = torch.device('cuda:%d'%(0))
im_size = 512

net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=im_size)#, big=args.big )
net_ig.to(device)

epoch = 100000
ckpt = './models/all_%d.pth'%(epoch)

ckpt = '../good_art_1k_512/models/all_%d.pth'%(50000)

checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
net_ig.load_state_dict(checkpoint['g'])
load_params(net_ig, checkpoint['g_ema'])
net_ig.eval()

net_id = Discriminator(ndf=64, im_size=im_size)
net_id.load_state_dict(checkpoint['d'])

noise = torch.randn(8, noise_dim).to(device)
g_imgs = net_ig(noise)[0]

vutils.save_image(g_imgs.add(1).mul(0.5), 
                        os.path.join('./', '%d.png'%(1)))

@torch.no_grad()
def get_feat(net_ig, input):
    feat_4   = net_ig.init(input)
    feat_8   = net_ig.feat_8(feat_4)
    feat_16  = net_ig.feat_16(feat_8)
    feat_32  = net_ig.feat_32(feat_16)
    return feat_32

@torch.no_grad()
def gen_image(net_ig, input):
    feat_4   = net_ig.init(input)
    feat_8   = net_ig.feat_8(feat_4)
    feat_16  = net_ig.feat_16(feat_8)
    feat_32  = net_ig.feat_32(feat_16)

    feat_32[1,1:10] = 0

    feat_64  = net_ig.se_64( feat_4, net_ig.feat_64(feat_32) )
    feat_128 = net_ig.se_128( feat_8, net_ig.feat_128(feat_64) )
    feat_256 = net_ig.se_256( feat_16, net_ig.feat_256(feat_128) )
    feat_256[:,1:10] = 0
    feat_512 = net_ig.se_512( feat_32, net_ig.feat_512(feat_256) )
    return net_ig.to_big(feat_512), feat_32

g_imgs, feat_32 = gen_image(net_ig, noise)

vutils.save_image(g_imgs[1].add(1).mul(0.5), 
                        os.path.join('./', '%d_mod.png'%(1)))


feat_32 = get_feat(net_ig, noise)
im = feat_32[1].unsqueeze(1)#.mean(0).unsqueeze(0)
vutils.save_image(im, 
    os.path.join('./', 'fm_%d_2.png'%(32)),
    normalize=True)

w = net_ig.init.init[0].weight[0][:8].unsqueeze(1)
vutils.save_image(F.interpolate(w, 128), 
    os.path.join('./', 'w_%d.png'%(4)),
    normalize=True)

for i in range(3,9):
    s = 2**i
    w = getattr(net_ig, 'feat_%d'%s)[1].weight
    print(s, 'std: ', w.std().item(), 'mean: ', w.mean().item(), '\n')
    
    #print('diff: ', F.l1_loss(w[0],w[1]).item())

for i in range(3,9):
    s = 2**i
    m = getattr(net_ig, 'feat_%d'%s)
    if len(m)==4:
        w = m[2].weight
    else:
        w = m[3].weight
    print(s, 'std: ', w.std().item(), 'mean: ', w.mean().item(), '\n')
    #print('diff: ', F.l1_loss(w[0],w[1]).item())



for i in range(2,7):
    s = 2**i
    w = getattr(net_id, 'down_%d'%s).main[0].weight
    print('std: ', w.std().item())
    print('diff: ', F.l1_loss(w[0],w[1]).item())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)

    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = './models/%d.pth'%(epoch)
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = 'eval_%d'%(epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = torch.randn(args.batch, noise_dim).to(device)
                g_imgs = net_ig(noise)[0]
                g_imgs = F.interpolate(g_imgs, 512)
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
