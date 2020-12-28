import argparse
import os
import numpy as np
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.autograd import Variable

from data.datasets import *
from models.Generator import *
from models.function import *


os.makedirs("testresult", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Multipie_normal_frontal", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--imsize", type=int, default=128, help="image size")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)


# Parameterize generator
cuda = torch.cuda.is_available()

generator = GeneratorResNet()

if cuda:
    generator = generator.cuda()

generator.load_state_dict(torch.load("saved_models/generator_20.pth"))


# Dataloader
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
imsize = opt.imsize


lr_transforms = [transforms.Resize((imsize//8, imsize//8)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]

hr_transforms = [transforms.Resize((imsize, imsize)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                 ]

hr_transforms2 = [transforms.Resize((imsize*2, imsize*2)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                 ]


loader = DataLoader(CustomDataset('data/testingset.txt',
                                  lr_transforms=lr_transforms,
                                  hr_transforms=hr_transforms,
                                  hr_transforms2=hr_transforms2),
                    batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


loader2 = DataLoader(ImageDataset("../../Data/%s" % opt.dataset_name,
                                   lr_transforms=lr_transforms,
                                   hr_transforms=hr_transforms,
                                   hr_transforms2=hr_transforms2),
                     batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


# ------------------------------------------------------------------------------------------
#  Testing
# ------------------------------------------------------------------------------------------

for i, imgs in enumerate(loader):
        # Configure model input
        imgs_lr = imgs['lrc'].cuda()
        imgs_hrgt = imgs['hrgt'].cuda()
        imgs_hr256 = imgs['hrgt_256'].cuda()

        imgs_lr_up = nn.functional.interpolate(imgs_lr, scale_factor=8)

        # Dataloader 2
        for j, imgs in enumerate(loader2):
            imgs_lrs = imgs['lrs'].cuda()
            imgs_hrsgt = imgs['hrs'].cuda()
            imgs_hrs256 = imgs['hrs256'].cuda()


        # Generate a high resolution image from low resolution input
        out_fc, out_fs, heatmap_fc, heatmap_fs = generator(imgs_lr, imgs_lrs)


        # Save CPGAN outputs
        batches_done = i
        save_image(out_fc.data, 'testresult/%04d.png' % batches_done, normalize=True)

        
         
       
                      
