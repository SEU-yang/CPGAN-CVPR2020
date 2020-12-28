import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19


### Define STN
class SpatialTransformer1(nn.Module):
   
    def __init__(self, in_channels = 128, spatial_dims = (32, 32), kernel_size = 3, use_dropout=False):
        super(SpatialTransformer1, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 
        self._ksize = kernel_size
        self.dropout = use_dropout
      
        # localization net 
        
        self.conv1 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(128, 20, kernel_size=self._ksize, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x12x12]
        self.conv2 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(20, 20, kernel_size=self._ksize, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x2x2]
                     
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(20*2*2, 20),
            nn.ReLU(True),
            nn.Linear(20, 2 * 3)          
            )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

     
    def forward(self, x):                 
        x1 = self.conv1(x)    
        x1 = self.conv2(x1)  
        #print(x1.shape)
        x1 = x1.view(-1, 20*2*2)         
        x1 = self.fc_loc(x1)  
        x1 = x1.view(-1, 2, 3) # change it to the 2x3 matrix #1*2*3
        #print(x.size())
        affine_grid_points = F.affine_grid(x1, torch.Size((x.size(0), self._in_ch, self._h, self._w)))  # 1,2,2  * 1,512,32,32= 1,32,32,2
        #print(batch_images.size(0))
        #print(affine_grid_points.size(0))
        
        assert(affine_grid_points.size(0) == x.size(0)) #"The batch sizes of the input images must be same as the generated grid."  1=1
        rois = F.grid_sample(x, affine_grid_points)   # 1,512,32,32 * 1,32,32,2 = 1,512,32,32
        #print("rois found to be of size:{}".format(rois.size()))
        return rois   ## 1,128,32,32

        
class SpatialTransformer2(nn.Module):
   
    def __init__(self, in_channels = 64, spatial_dims = (64, 64),use_dropout=False):
        super(SpatialTransformer2, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 
        #self._ksize = kernel_size
        self.dropout = use_dropout

         # localization net 
        
        self.conv1 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(64, 128, 5, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x128x28x28]
        self.conv2 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(128, 20, 5, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x10x10]

        self.conv3 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(20, 20, 3, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x3x3]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1*20*3*3, 20),
            nn.ReLU(True),
            nn.Linear(20, 2 * 3) )
                   

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        

    def forward(self, x):                 
        x1 = self.conv1(x)    
        x1 = self.conv2(x1)   
        x1 = self.conv3(x1)   
       
        x1 = x1.view(-1, 20*3*3)
        
        x1 = self.fc_loc(x1)  
     
        x1 = x1.view(-1, 2, 3) # change it to the 2x3 matrix #1*2*3
        #print(x.size())
        affine_grid_points = F.affine_grid(x1, torch.Size((x.size(0), self._in_ch, self._h, self._w)))  # 1,2,2  * 1,256,64,64 = 1,64,64,2
        #print(batch_images.size(0))
        #print(affine_grid_points.size(0))
        
        assert(affine_grid_points.size(0) == x.size(0)) #"The batch sizes of the input images must be same as the generated grid."  1=1
        rois = F.grid_sample(x, affine_grid_points)   # 1,256,64,64 * 1,64,64,2 = 1,256,64,64
        #print("rois found to be of size:{}".format(rois.size()))
        return rois  ## 1,256,64,64
    


class SpatialTransformer3(nn.Module):
   
    def __init__(self, in_channels = 128, spatial_dims = (128, 128),use_dropout=False):
        super(SpatialTransformer3, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 
        #self._ksize = kernel_size
        self.dropout = use_dropout

         # localization net 
        
        self.conv1 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(in_channels, 64, 5, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x64x60x60]
        self.conv2 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(64, 32, 3, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x28x28]

        self.conv3 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(32, 20, 5, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x10x10]

        self.conv4 = nn.Sequential(nn.MaxPool2d(2,2),
                     nn.Conv2d(20, 20, 3, stride=1, padding=0, bias=False),
                     nn.ReLU()) # size : [1x20x3x3]
                     
                     
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1*20*3*3, 20),
            nn.ReLU(True),
            nn.Linear(20, 2 * 3))
                    
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1 ,0], dtype=torch.float))
       

    def forward(self, x):        
        x1 = self.conv1(x)    
        x1 = self.conv2(x1)   
        x1 = self.conv3(x1)    
        x1 = self.conv4(x1)    
        
        x1 = x1.view(-1, 20*3*3)
        
        x1 = self.fc_loc(x1)  
     
        x1 = x1.view(-1, 2, 3) # change it to the 2x3 matrix #1*2*3
        #print(x.size())
        affine_grid_points = F.affine_grid(x1, torch.Size((x.size(0), self._in_ch, self._h, self._w)))  # 1,2,3  * 1,128,128,128 = 1,64,64,2
        #print(batch_images.size(0))
        #print(affine_grid_points.size(0))
        
        assert(affine_grid_points.size(0) == x.size(0)) #"The batch sizes of the input images must be same as the generated grid."  1=1
        rois = F.grid_sample(x, affine_grid_points)  
        #print("rois found to be of size:{}".format(rois.size()))
        return rois  ## 1,128,128,128


      
## define attention model

## define attention model

class Self_Attn(nn.Module):
    #""" Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        ##"""
        #    inputs :
        #        x : input feature maps( B X C X W X H)
        #    returns :
        #        out : self attention value + input feature 
        #        attention: B X N X N (N is Width*Height)
        #"""
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention


        
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__() 
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

        
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features),
                        nn.ReLU(),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
        #self.attn = Self_Attn(in_features, 'relu')
    
    def forward(self, x):
        res= x + self.conv_block(x)        
        return res


class MyResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(MyResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features),
                        nn.ReLU(),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        res= x + self.conv_block(x)      
        return res


                    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x + x * y


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        self.soca=SOCA(in_feat, reduction=reduction)
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]


        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat



# dlib 68 landmark predict   
#~ detector = dlib.get_frontal_face_detector() 
#~ predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 


def get_facial_landmarks(img):  
    rects = detector(img, 0) 
    #print(rects.shape)  
    rect = rects[0]
    #~ print(len(rects))
    #~ print("22222")
    shape = predictor(img, rect)
    a=np.array([[pt.x, pt.y] for pt in shape.parts()])    
    b=a.astype('float').reshape(-1,136)
    return b 
    
    
def _putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y
        grid_x = crop_size_x
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        #start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx 
        yy = yy 
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap
        
        
def _putGaussianMaps(keypoints,crop_size_y, crop_size_x, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = _putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img


