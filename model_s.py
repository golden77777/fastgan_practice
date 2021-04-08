import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import random
import math
seq = nn.Sequential

import torchgeometry as tgm
gauss = tgm.image.GaussianBlur((5, 5), (2, 2))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        nn.BatchNorm2d( channel*2 ),
                        GLU()
                        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


class HLayer(nn.Module):
    def __init__(self, channel, nz=16):
        super().__init__()

        self.nz = nz
        self.from_noise = nn.Sequential(
                        conv2d(nz, channel//2, 1, 1, 0, bias=False),
                        nn.BatchNorm2d( channel//2 ), nn.GELU(),
                        conv2d(channel//2, channel//2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d( channel//2 ), nn.GELU(),
                        conv2d(channel//2, channel, 3, 1, 1, bias=False),
                        nn.BatchNorm2d( channel ), nn.GELU(),
                        conv2d(channel, channel, 1, 1, 0, bias=False),
                        nn.BatchNorm2d( channel ), nn.GELU(),
                        )

        self.merge = nn.Sequential(
                        conv2d(channel*2, channel*2, 1, 1, 0, bias=False),
                        nn.BatchNorm2d( channel*2 ), nn.GELU(),
                        conv2d(channel*2, channel*2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d( channel*2 ), nn.GELU(),
                        conv2d(channel*2, channel*2, 1, 1, 0, bias=False),
                        nn.BatchNorm2d( channel*2 ), 
                        GLU(),
                        )

    def forward(self, feat):
        noise = torch.randn(feat.shape[0], self.nz, feat.shape[2], feat.shape[3], device=feat.device)
        noise_feat = self.from_noise(noise)
        feat_hat = self.merge(torch.cat([feat, noise_feat], dim=1))
        return feat + feat_hat


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2),
        GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2),
        GLU(),
        )
    return block


def UpBlockH(in_planes, out_planes):
    block = nn.Sequential(
        convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2),
        GLU(),
        HLayer(out_planes)
        )
    return block


from random import randint
class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256, sle=True, big=False):
        super(Generator, self).__init__()

        self.sle = sle

        nfc_multi = {4:8, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])#, groups=nfc[8]//4)
        self.feat_16 = UpBlockComp(nfc[8], nfc[16])#, groups=nfc[16]//4)
        self.feat_32 = UpBlockComp(nfc[4], nfc[32])#, groups=1)
        self.feat_64 = UpBlockComp(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlockComp(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_256 = conv2d(nfc[256], nc, 3, 1, 1, bias=False) #, nn.Tanh() )
        
        self.to_1024 = seq(
            nn.UpsamplingNearest2d(scale_factor=2),
            conv2d(nfc[256], nfc[512], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nfc[512]), nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            conv2d(nfc[512], nfc[1024], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nfc[1024]), nn.ReLU(),
            conv2d(nfc[1024], 3, 3, 1, 1, bias=False)#, nn.Tanh(),
        )

    def forward(self, z):

        feat_4  = 0.4 * self.init(z)

        feat_8  = 0.4 * self.feat_8(feat_4)
        feat_16 = 0.6 * self.feat_16(feat_8)
        feat_32 = 0.8 * self.feat_32(feat_16)

        feat_64 = self.feat_64(feat_32)
        feat_64 = self.se_64(feat_4, feat_64)

        feat_128 = self.feat_128(feat_64)
        feat_128 = self.se_128(feat_8, feat_128)
        
        feat_256 = self.feat_256(feat_128)
        feat_256 = self.se_256(feat_16, feat_256)

        img_256 = self.to_256(feat_256)
        img_1024 = self.to_1024(feat_256) + gauss(F.interpolate(img_256, size=1024, mode='bilinear'))
        return [torch.tanh(img_1024),torch.tanh(img_1024)]

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat) 


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2))
            
        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),)

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 1.2


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, sle=True, decode=True, big=False):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        self.sle = sle
        self.decode = decode

        self.down_256 = nn.Sequential( 
                                    conv2d(nc, ndf//8, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(ndf//8, ndf//4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf//4),
                                    nn.LeakyReLU(0.2, inplace=True), )

        self.down_64 = DownBlockComp(ndf//4, ndf//2)
        self.down_32 = DownBlockComp(ndf//2, ndf*1)
        self.down_16 = DownBlockComp(ndf*1,  ndf*2)
        self.down_8  = DownBlockComp(ndf*2,  ndf*4)
        self.down_4  = DownBlockComp(ndf*4,  ndf*8)

        self.rf_big = nn.Sequential(
                            DownBlock(ndf*8,  ndf*16),
                            conv2d(ndf * 16, 1, 4, 1, 0, bias=False))

        self.down_from_small = nn.Sequential( conv2d(nc, ndf//2, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(ndf//2,  ndf*1),
                                            DownBlock(ndf*1,  ndf*2),
                                            DownBlock(ndf*2,  ndf*4), )
        self.rf_small = nn.Sequential(
                            DownBlock(ndf*4,  ndf*8),
                            conv2d(ndf*8, 1, 4, 1, 0, bias=False))

        if self.decode:
            self.decoder_big = SimpleDecoder(ndf*4, nc)
            self.decoder_small = SimpleDecoder(ndf*4, nc)
        
    def forward(self, img, label='fake'):
        if type(img) == list:
            img_128 = F.interpolate(img[0], 128, mode='bilinear', align_corners=True)
            img = img[0]
        else:
            img_128 = F.interpolate(img, 128, mode='bilinear', align_corners=True)
            
        feat_128 = self.down_256(img)        
        feat_64 = self.down_64(feat_128)
        feat_32 = self.down_32(feat_64)
        feat_16 = self.down_16(feat_32)
        feat_8 = self.down_8(feat_16)
        feat_4 = self.down_4(feat_8)

        rf_0 = self.rf_big(feat_4).view(-1)

        feat_small = self.down_from_small(img_128)
        rf_1 = self.rf_small(feat_small).view(-1)

        
        if label=='real' and self.decode:    
            rec_img_big = self.decoder_big(feat_8)
            rec_img_small = self.decoder_small(feat_small)

            return torch.cat([rf_0, rf_1], dim=0) ,[rec_img_big, rec_img_small,rec_img_small], 1
        
        return torch.cat([rf_0, rf_1], dim=0) 


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_planes*2),
                GLU()
            )        
        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False) )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)