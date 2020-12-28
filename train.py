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


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="Multipie_normal_frontal", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--imsize", type=int, default=128, help="image size")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)


cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()


# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
FAN_heatmap = FAN_heatmap()


if cuda:
    generator = generator.cuda()
    FAN_heatmap = FAN_heatmap.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

initialize_weights(generator)
initialize_weights(discriminator)


if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Dstaloader
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


loader = DataLoader(CustomDataset('data/trainsetall.txt',
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
#  Training
# ------------------------------------------------------------------------------------------

for epoch in range(opt.epoch, opt.n_epochs):

    adjust_learning_rate(optimizer_G, epoch, lrr=1e-4)
    adjust_learning_rate(optimizer_D, epoch, lrr=1e-4)

    ## Dataloader
    for i, imgs in enumerate(loader):

        print("Current batch : {}".format(i))

        # Configure model input
        imgs_lr = imgs['lrc'].cuda()
        imgs_hrgt = imgs['hrgt'].cuda()
        imgs_hr256 = imgs['hrgt_256'].cuda()

        imgs_lr_up = nn.functional.interpolate(imgs_lr, scale_factor=8)


        ## Dataloader 2
        for j, imgs in enumerate(loader2):
            imgs_lrs = imgs['lrs'].cuda()
            imgs_hrsgt = imgs['hrs'].cuda()
            imgs_hrs256 = imgs['hrs256'].cuda()


        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------------------------------------------------------------------------------------------------
        #  Train Generators
        # ------------------------------------------------------------------------------------------------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        out_fc, out_fs, heatmap_fc, heatmap_fs = generator(imgs_lr, imgs_lrs)

        ## Content face loss
        ## Style loss
        loss_style = calc_style_loss(out_fc, imgs_hrsgt)

        # vgg loss
        face_C_features = feature_extractor(out_fc)  # 128

        real_C_features = Variable(feature_extractor(imgs_hrgt).data, requires_grad=False)

        loss_content_c = criterion_content(face_C_features, real_C_features)

        # l2 loss
        loss_G2_c = criterion_GAN(out_fc, imgs_hrgt)

        # landmark loss
        real_heatmaps = FAN_heatmap(imgs_hr256)

        loss_land_c = criterion_GAN(heatmap_fc, real_heatmaps) / 68

        # Adversarial loss
        gen_validity = discriminator(out_fc)

        loss_GAN = criterion_GAN(gen_validity, valid)

        # Total loss
        loss_G_c = (1e-2) * loss_style + loss_G2_c + (1e-1) * loss_GAN + (1e-2) * loss_content_c + 10 * loss_land_c


        ## Style face loss
        # vgg loss
        face_S_features = feature_extractor(out_fs)  # 128

        real_S_features = Variable(feature_extractor(imgs_hrsgt).data, requires_grad=False)

        loss_content_s = criterion_content(face_S_features, real_S_features)

        # l2 loss
        loss_G2_s = criterion_GAN(out_fs, imgs_hrsgt)

        # landmark loss
        real_S_heatmaps = FAN_heatmap(imgs_hrs256)

        loss_land_s = criterion_GAN(heatmap_fs, real_S_heatmaps) / 68

        # ST total loss
        loss_G_s = loss_G2_s + (1e-2) * loss_content_s + 10 * loss_land_s

        # Total loss
        loss_G = loss_G_c + loss_G_s

        loss_G.backward()
        optimizer_G.step()

        # ------------------------------------------------------------------------------------------------------------
        #  Train Discriminator
        # ------------------------------------------------------------------------------------------------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hrgt), valid)
        loss_fake = criterion_GAN(discriminator(out_fc.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # ------------------------------------------------------------------------------------------------------------
        #  Log Progress
        # ------------------------------------------------------------------------------------------------------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(loader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(loader) + i

        # with open("my_log.txt", 'a') as f:
        #     f.write(curr_progress + "\n")

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and CPGAN outputs
            save_image(torch.cat((imgs_lr_up.data, out_fc.data, imgs_hrgt.data, imgs_hrsgt.data), -2),
                       'images/%d.png' % batches_done, normalize=True)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
