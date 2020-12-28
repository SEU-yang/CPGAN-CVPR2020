import torch.nn as nn
import torch

from models.base_model import *
from models.function import *
from models.MyFAN import *


class Internal_cpnet(nn.Module):
    def __init__(self):
        super(Internal_cpnet, self).__init__()
                  
        # Residual blocks
        res_blocks_fc = []
        for _ in range(2):
            res_blocks_fc.append(ResidualBlock(256))
        self.res_blocks_fc = nn.Sequential(*res_blocks_fc)               
        
        
        # Channel attention               
        self.channelatt = CALayer(256, reduction=16) 
        
        # Copy block 
        self.non_local = Self_Attn(256, 'relu')
                             
        
    def forward(self, x):       
        x_fc_res = self.res_blocks_fc(x)
        x_fc_ca = self.channelatt(x_fc_res)
        x_nonlocal, attention = self.non_local(x_fc_ca)
        return x_nonlocal, attention
        


class External_cpnet(nn.Module):
    def __init__(self):
        super(External_cpnet, self).__init__()
        
        # Residual blocks
        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # External copy block
        self.sanet = SANet(in_dim=128)
        
        
    def forward(self, fc, fs):                    

        fc_res = self.res_blocks(fc)
        fs_res = self.res_blocks(fs)

        fcs, out_s = self.sanet(fc_res, fs_res)

        return fcs, out_s, fs_res
    


class GeneratorResNet(nn.Module):
    def __init__(self):
        super(GeneratorResNet, self).__init__()
        # Input feature extration conv layer
        self.conv0 = nn.Sequential(nn.Conv2d(3, 256, 5, 1, 2),
                        SwitchNorm2d(256),
                        nn.ReLU())
    
        # Internal copy & paste network                  
        self.in_cpnet = Internal_cpnet()       
                
        # Feature Shrinking
        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                        SwitchNorm2d(128),
                        nn.ReLU()
                        )
        # Spatial Transform Network
        self.stn = SpatialTransformer1()

        # External copy & paste network                                                   
        self.ex_cpnet = External_cpnet()
               
         # Upsampling layers
        upsampling = []
        for out_features in range(1):
            upsampling += [nn.Conv2d(128, 512, 3, 1, 1),
                           SwitchNorm2d(512),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*upsampling)

        # Landmark heatmap prediction                
        self.fan = MyFAN()
        
        # Concat face feature maps and landmark heatmaps
        self.conv2 = nn.Sequential(nn.Conv2d(196, 128, 3, 1, 1),
                       SwitchNorm2d(128),
                       nn.ReLU())

        # Residual blocks
        res_blocks_out = []
        for _ in range(1):
            res_blocks_out.append(ResidualBlock(128))
        self.res_block_out = nn.Sequential(*res_blocks_out)

        # output conv layer
        self.conv3 = nn.Sequential(nn.Conv2d(128, 32, 5, 1, 2),
                        SwitchNorm2d(32),
                        nn.ReLU()
                       )

        # output conv layer
        self.conv4 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())               
                              
              
    def forward(self, fc, fs):

            x_fc_in = self.conv0(fc)         ## input content features 256*16*16
            x_fs_in = self.conv0(fs)         ## input style features 256*16*16

            x_incp, attention_in = self.in_cpnet(x_fc_in)  ## internal CPnet  256*16*16

            x_incp_FS = self.conv1(x_incp)   ## shrunken content features 128*16*16
            x_fs_FS = self.conv1(x_fs_in)    ## shrunken style features 128*16*16

            a_16 = x_incp_FS
            b_16 = x_fs_FS

            a_16_en, b_16_att, b_16_en = self.ex_cpnet(a_16, b_16)    ## external CPnet1  128*16*16

            a_32 = self.up(a_16_en)                      ## upsampled content features 128*32*32
            b_32 = self.up(b_16_en)                      ## upsampled style features 128*32*32

            a_32_en, b_32_att, b_32_en = self.ex_cpnet(a_32, b_32)    ## external CPnet2  128*32*32

            a_64 = self.up(a_32_en)                      ## upsampled content features 128*64*64
            b_64 = self.up(b_32_en)                      ## upsampled style features 128*64*64

            heatmap_fc = self.fan(a_64)               ## facial landmark heatmaps of content faces  68*64*64
            heatmap_fs = self.fan(b_64)               ## facial landmark heatmaps of style faces  68*64*64

            gen_fan_fc = torch.cat(heatmap_fc,0)
            gen_fan_fs = torch.cat(heatmap_fs,0)     

            fc_64_concat = torch.cat([a_64, gen_fan_fc], 1)  ## concatenates content features 196*64*64
            fs_64_concat = torch.cat([b_64, gen_fan_fs], 1)   ## concatenates style features 196*64*64

            a_64 = self.conv2(fc_64_concat)    ## shrunken content features 128*64*64
            b_64 = self.conv2(fs_64_concat)    ## shrunken style features 128*64*64

            a_64_en, b_64_att, b_64_en = self.ex_cpnet(a_64, b_64)   ## external CPnet3  128*64*64

            a_128 = self.up(a_64_en)       ## upsampled content features 128*128*128
            b_128 = self.up(b_64_en)        ## upsampled style features 128*128*128

            fcout = self.res_block_out(a_128)   ## enhanced content features 128*128*128
            fsout = self.res_block_out(b_128)   ## enhanced style features 128*128*128

            out_fc_f = self.conv3(fcout)       ## output content features 32*128*128
            out_fs_f = self.conv3(fsout)       ## output style features 32*128*128

            out_fc = self.conv4(out_fc_f)      ## output content features 3*128*128
            out_fs = self.conv4(out_fs_f)     ## output style features 3*128*128

            return out_fc, out_fs, gen_fan_fc, gen_fan_fs



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)



