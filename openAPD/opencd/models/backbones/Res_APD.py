from mmseg.models.builder import BACKBONES
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, build_upsample_layer)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmseg.models.backbones import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import cat
import torch.utils.checkpoint as cp
from .gcn_lib import AlignGrapher
from timm.models.vision_transformer import PatchEmbed, Block
from .visualize import visualize_feature
import numpy as np
import random
import time
from torch.nn.functional import normalize

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_channels))
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.relu(out)
        out = self.block2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class PerturbMask(nn.Module):

    def __init__(self, p):
        super(PerturbMask, self).__init__()
        self.p = p

    def forward(self, x1):
        N, c, h, w = x1.shape
        
        zero_map = torch.arange(c) % self.p == 0
        map_mask = zero_map.unsqueeze(0).expand((N, -1))
 
        out_x1 = torch.zeros_like(x1)
        out_x1[~map_mask, ...] = x1[~map_mask, ...]
        
        return out_x1


class PerturbExchange(nn.Module):

    def __init__(self, p=2):
        super(PerturbExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        negative_map = torch.arange(c) % self.p == 0
        exchange_mask = negative_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2

eps = 1.0e-5
class PM(nn.Module): #Perturbation Module
    def __init__(self, in_planes,p):
        super(PM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.exchange = PerturbExchange(p)
        self.mask = PerturbMask(4)
    def forward(self, input1, input2):
        input1, input2 = self.exchange(input1, input2)
        diff = torch.sub(input1, input2)
        diff_temp = self.mask(diff)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff_temp))))
        ori_out = self.fc2(self.relu1(self.fc1(diff_temp)))
        att = self.sigmoid(avg_out + ori_out)
        feature1 = input1 * att + input1
        feature2 = input2 * att + input2
        different = torch.sub(feature1, feature2)
        return feature1, feature2, different,att

@BACKBONES.register_module()      
class SiaResAPD_18(nn.Module):

    def __init__(self,in_channel=3,block=BasicBlock, num_block=[2, 2, 2, 2],**kwargs):
        super(SiaResAPD_18,self).__init__()
    
        self.in_channels = 64
        self.blockhead_channels = 64
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size = 7, stride = 2, padding = 3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)
        self.drop5 = nn.Dropout2d(0.2)
        self.drop6 = nn.Dropout2d(0.2)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.drop7 = nn.Dropout2d(0.1)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.drop8 = nn.Dropout2d(0.1)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv10 = nn.Conv2d(64, 64, 1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        
        self.convN = nn.Conv2d(256, 256, 3, 2, 1)
        self.convN2 =nn.Conv2d(512, 256, 3, 1, 1)
        
        self.att3 = AlignGrapher(128,64,8,8)
        self.att2 = AlignGrapher(256,32,4,4)
        self.att = AlignGrapher(512,16,2,2)

        self.ede2 = PM(128,2)
        self.ede3 = PM(256,2)
        self.ede4 = PM(512,2)
        
        self.relu = nn.ReLU()
        self.latlayerdiff3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.latlayerdiff4 = nn.ConvTranspose2d(256, 64, 4, stride=4)
        self.latlayerdiff5 = nn.ConvTranspose2d(512, 64, 8, stride=8)
        self.maskconv = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.blockhead_channels != block.expansion*out_channels:
            downsample = nn.Sequential(nn.Conv2d(self.blockhead_channels, out_channels*block.expansion, stride=stride, kernel_size=1,bias=False),
                                            nn.BatchNorm2d(block.expansion*out_channels))
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.blockhead_channels, out_channels, downsample=downsample, stride=stride))
            # only add the downsample layer for the head basic block
            downsample = None
            self.blockhead_channels = out_channels*block.expansion
        return nn.Sequential(*layers)
    
    def _upsample_add(self, x, y):
          _,_,H,W = y.size()
          return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def forward(self, x,t):
        #x,t = inputs
        #encoder
        conv1_x = self.conv1(x)
        conv1_t = self.conv1(t)
        
        temp_x = self.maxpool(conv1_x)
        temp_t = self.maxpool(conv1_t)        
        diff1 = temp_x -temp_t
        
        conv2_x = self.conv2(temp_x)
        conv2_t = self.conv2(temp_t)
        diff2 = conv2_x - conv2_t
        
        conv3_x = self.conv3(conv2_x)
        conv3_t = self.conv3(conv2_t)
        conv3_x, conv3_t = self.att3(conv3_x,conv3_t)
        conv3_x , conv3_t ,diff3,att_3 = self.ede2(conv3_x , conv3_t)
        
        conv4_x = self.conv4(conv3_x)
        conv4_t = self.conv4(conv3_t)
        conv4_x, conv4_t = self.att2(conv4_x,conv4_t)
        conv4_x , conv4_t ,diff4,att_4 = self.ede3(conv4_x, conv4_t)#256

        conv5_x = self.conv5(conv4_x)
        conv5_t = self.conv5(conv4_t)
        conv5_x, conv5_t = self.att(conv5_x,conv5_t)
        conv5_x , conv5_t ,diff5,att_5 = self.ede4(conv5_x, conv5_t)#512
        
        diff_3 = self.latlayerdiff3(att_3)
        diff_4 = self.latlayerdiff4(att_4)
        diff_5 = self.latlayerdiff5(att_5)
        diff_temp_final = self.maskconv(diff_3 + diff_4 + diff_5)
        
        C5 = torch.cat((conv5_x, conv5_t), dim=1)#1024
        #decoupled-decoder
        d5 = self.drop5(C5)
        up_6 = self.up6(d5) #512
        merge6 = torch.cat([up_6, diff4], dim=1) #512+256
        c6 = self.dconv_up3(merge6)#256
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, diff3], dim=1)
        c7 = self.dconv_up2(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, diff2], dim=1)
        c8 = self.dconv_up1(merge8)
        d8 = self.drop8(c8) 
        c9 = self.conv10(d8)
        
        # top-down
        p5 = self.latlayer4(diff5)
        
        p4 = self.latlayer3(diff4)
        p4 = self._upsample_add(p5, p4)
        
        p3 = self.latlayer2(diff3)
        p3 = self._upsample_add(p4, p3)
        
        p2 = self.latlayer1(diff2)
        p2 = self._upsample_add(p3, p2)
        
        # down-top
        N2 = p2
        N2_ = self.convN(N2)
        N2_ = self.relu(N2_)
 
        N3 = N2_ + p3
 
        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)
        N4 = N3_ + p4
        
        N4_ = self.convN(N4)
        N4_ = self.relu(N4_)
        N5 = N4_ + p5

        N2 = N2 + p2
        N3 = N3 + p3
        N4 = N4 + p4
        N5 = N5 + p5
        
        c2 = torch.cat((N2,p2), dim=1)
        c3 = torch.cat((N3,p3), dim=1)
        c4 = torch.cat((N4,p4), dim=1)
        c5 = torch.cat((N5,p5), dim=1)
        
        c2 = self.convN2(c2)
        c3 = self.convN2(c3)
        c4 = self.convN2(c4)
        c5 = self.convN2(c5)  
        
        conv5_x_temp = conv5_x
        conv5_t_temp = conv5_t
        conv5_x_temp = conv5_x_temp.unsqueeze(4)
        conv5_t_temp = conv5_t_temp.unsqueeze(4)
        distance5 = F.pairwise_distance(conv5_x_temp, conv5_t_temp, p=2)#[8,512,16,16]
        distance5 = distance5/torch.max(distance5)
        distance5 = self.latlayerdiff5(distance5)
        
        conv3_x_temp = conv3_x
        conv3_t_temp = conv3_t
        conv3_x_temp = conv3_x_temp.unsqueeze(4)
        conv3_t_temp = conv3_t_temp.unsqueeze(4)
        distance3 = F.pairwise_distance(conv3_x_temp, conv3_t_temp, p=2)
        distance3 = distance3/torch.max(distance3)
        distance3 = self.latlayerdiff3(distance3)
        
        conv4_x_temp = conv4_x
        conv4_t_temp = conv4_t
        conv4_x_temp = conv4_x_temp.unsqueeze(4)
        conv4_t_temp = conv4_t_temp.unsqueeze(4)
        distance4 = F.pairwise_distance(conv4_x_temp, conv4_t_temp, p=2)
        distance4 = distance4/torch.max(distance4)
        distance4 = self.latlayerdiff4(distance4)
        distance = self.maskconv(distance3+distance4+distance5)
        
        x = []
        x.append(c2)
        x.append(c3)
        x.append(c4)
        x.append(c9)
        x.append(diff_temp_final)
        x.append(distance)
        return x
    