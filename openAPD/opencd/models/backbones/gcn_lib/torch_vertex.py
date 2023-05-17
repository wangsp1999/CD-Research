# 2022.06.17-Changed for building ViG model
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        #print("max_value:")
        #print(max_value.shape)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x,y, relative_pos=None):
        B, C, H, W = x.shape
        if self.r > 1:
            z = F.avg_pool2d(y, self.r, self.r)
            z = z.reshape(B, C, -1, 1).contiguous()             
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, z, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, z)
        return x.reshape(B, -1, H, W).contiguous()
        
class C_Aggregation(nn.Module):
    def __init__(self, channels, patch_size=16):
    
        super(C_Aggregation, self).__init__()
        self.patchsize = patch_size
        embed_dim = patch_size * patch_size * channels
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(channels,embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H0, W0 = x.shape
        imgsize = H0
        patch_num = imgsize // self.patchsize
        patch_seq = [i * patch_num + j for i in range(1,patch_num - 1) for j in range(1,patch_num)]
        x = F.pad(x, (self.patchsize, self.patchsize, self.patchsize, self.patchsize), "constant", 0)
        B, C, H, W = x.shape
        x_temp = self.conv1(x)
        B1, C1, H1, W1 = x_temp.shape
        x_temp = x_temp.reshape(B1,C1,-1).contiguous()
        x_final = x_temp
        for idx in patch_seq:
          out = x_temp[:, :, idx - patch_num] + x_temp[:, :, idx + patch_num] + x_temp[:, :, idx - 1] + x_temp[:, :, idx + 1] + x_temp[:, :, idx - patch_num -1] + x_temp[:, :, idx - patch_num + 1] + x_temp[:, :, idx + patch_num -1] + x_temp[:, :, idx + patch_num + 1]  
          out = out / 8
          x_final[:, :, idx] = out
        x_final = x_final.reshape(B1, -1, H1, W1).contiguous()
        x = x_final.reshape(B, C, H, W).contiguous()
        x = x[:, :, self.patchsize:imgsize+self.patchsize, self.patchsize:imgsize+self.patchsize]
        return x


class AlignGrapher(nn.Module):
    """
    AlignGrapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels,img_size,patch_size,r=1,kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, n=196, drop_path=0.0, relative_pos=False):
        super(AlignGrapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)
        self.caggregation = C_Aggregation(in_channels,patch_size)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x,y):
        _tmpX = x
        _tmpY = y
        
        x_faam = self.caggregation(x)
        y_faam = self.caggregation(y)
        x_faam = self.fc1(x_faam)
        y_faam = self.fc1(y_faam)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        xy_faam_final = self.graph_conv(x_faam,y_faam,relative_pos)
        xy_faam = self.fc2(xy_faam_final)
        x = self.drop_path(xy_faam) + _tmpX
        y = self.drop_path(xy_faam) + _tmpY
        return x,y