import torch
import torch.nn as nn
from models.FAN import FAN
from torch.utils.model_zoo import load_url



def adjust_learning_rate(optimizer, epoch, lrr):
    ##Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrr * (0.99 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)

class SANet(nn.Module):
    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)
        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(B, -1, H * W)
        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)
        H_Fs = self.h(style_feat).view(B, -1, H * W)
        out_s = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out_s = out_s.view(B, C, H, W)
        out = self.out_conv(out_s)
        out += content_feat
        return out, out_s



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



def calc_style_loss(input, target):
        assert (input.size() == target.size())        
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        mse_loss = nn.MSELoss()
        return mse_loss(input_mean, target_mean) + \
               mse_loss(input_std, target_std)


def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized  


        
class FAN_loss(nn.Module):
    def __init__(self):
        super(FAN_loss, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net
        self.criterion = nn.MSELoss()

    def forward(self, data, target):
        heat_predict = self.FAN_net(data)
        heat_gt = self.FAN_net(target)
        loss = self.criterion(self.FAN_net(data)[0], self.FAN_net(target)[0])
        #print(data[0].size())
        #print(target[0].size())
        # exit()
        return loss

class FAN_heatmap(nn.Module):
    def __init__(self):
        super(FAN_heatmap, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net        

    def forward(self, data):
        heat_gt = self.FAN_net(data) 
        #heatmap_gt = torch.cat(heat_gt,0)      
        return heat_gt[0]


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
