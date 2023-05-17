import cv2
import matplotlib.pyplot as plt
import os
import torch
import numpy as np


def visualize_feature(features: torch.Tensor, path='/disk3/xx/openAPD/opencd/models/backbones/', name='', sub_plot=False, cmap=None):
    """
    :param features: the feature map of your network layer, shape as C, W, H
    :param path: the image you need to save
    :param name: the file name to save
    :param sub_plot: if save sub plot
    :param cmap: camp
    :return: None
    """
    if features.dim() > 3 and features.shape[0] == 1:
        features = torch.squeeze(features, dim=0)
    num_img = features.shape[0]
    h = w = int(pow(num_img, .5))
    if h*w < num_img:
        h += 1
    if sub_plot:
        plt.figure(figsize=(100, 100))
        for i in range(num_img):
            # plt.imshow(features[i].detach().numpy())
            plt.subplot(h, w, i+1)
            plt.imshow(features[i].cpu().detach().numpy(), cmap=cmap)
        plt.savefig(os.path.join(path, name+'map.png'))
    plt.figure()
    if features.shape[0]==3:
        feature_sum = features.permute(1, 2, 0)
    else:
        feature_sum = features.cpu().sum(dim=0).squeeze()
    plt.imshow(feature_sum.cpu().detach().numpy(), cmap=cmap)
    plt.savefig(os.path.join(path, name+'map_sum.png'))
    return

