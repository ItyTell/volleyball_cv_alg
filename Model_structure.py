from __future__ import division

import os
from itertools import chain
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init_normal, parse_model_config


# layers 
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            if self.new_coords:
                x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
                x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh
            else:
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# function for adding layers 

# переделай чтоб функция сама добавляла модули в module_list

def convolution(module_id: int, in_channels: int, filters: int, size: int, stride: int, pad: int, batch_normalize: bool, activation: str) -> nn.Sequential:
    modules = nn.Sequential()
    modules.add_module("conv_{}".format(module_id), nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=size, stride=stride, padding=(pad, pad), bias=not batch_normalize))
    if batch_normalize:
        modules.add_module("batch_norm_{}".format(module_id), nn.BatchNorm2d(filters, momentum=0.1, eps=1e-05))
    if activation == "leaky":
        modules.add_module("leaky_{}".format(module_id), nn.LeakyReLU(0.1))
    return modules

def shortcut(module_id: int):
    modules = nn.Sequential()
    modules.add_module("shortcut_{}".format(module_id), nn.Sequential())
    return modules


def upsample(module_id: int, stride: int):
    modules = nn.Sequential()
    upsample = Upsample(scale_factor=stride, mode="nearest")
    modules.add_module("upsample_{}".format(module_id), upsample)
    return modules

def route(module_id: int):
    modules = nn.Sequential()
    modules.add_module("route_{}".format(module_id), nn.Sequential())
    return modules

def yolo(module_id: int, mask: List[int], anchors: List[Tuple[int, int]], num_classes: int):
    modules = nn.Sequential()
    anchors = [anchors[i] for i in mask]
    yolo_layer = YOLOLayer(anchors, num_classes, False)
    modules.add_module("yolo_{}".format(module_id), yolo_layer)
    return modules



def create_modules() -> Tuple[dict, nn.ModuleList]:
    hyperparams = {'type': 'net',
    'batch': 16,
    'subdivisions': 1,
    'width': 416,
    'height': 416,
    'channels': 3,
    'momentum': 0.9,
    'decay': 0.0005,
    'angle': '0',
    'saturation': '1.5',
    'exposure': '1.5',
    'hue': '.1',
    'learning_rate': 0.0001,
    'burn_in': 1000,
    'max_batches': 500200,
    'policy': 'steps',
    'steps': '400000,450000',
    'scales': '.1,.1',
    'optimizer': None,
    'lr_steps': [(400000, 0.1), (450000, 0.1)]}

    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()

    module = convolution(module_id=0, in_channels=output_filters[-1], filters=32, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(32)

    module = convolution(module_id=1, in_channels=output_filters[-1], filters=64, size=3, stride=2, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(64)

    module = convolution(module_id=2, in_channels=output_filters[-1], filters=32, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(32)

    module = convolution(module_id=3, in_channels=output_filters[-1], filters=64, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(64)

    module = shortcut(4)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=5, in_channels=output_filters[-1], filters=128, size=3, stride=2, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=6, in_channels=output_filters[-1], filters=64, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(64)

    module = convolution(module_id=7, in_channels=output_filters[-1], filters=128, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = shortcut(8)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=9, in_channels=output_filters[-1], filters=64, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(64)

    module = convolution(module_id=10, in_channels=output_filters[-1], filters=128, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = shortcut(11)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=12, in_channels=output_filters[-1], filters=256, size=3, stride=2, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=13, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=14, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(15)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=16, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=17, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(18)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=19, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=20, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(21)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=22, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=23, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(24)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=25, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=26, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(27)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=28, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=29, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(30)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=31, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=32, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(33)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=34, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=35, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = shortcut(36)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=37, in_channels=output_filters[-1], filters=512, size=3, stride=2, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=38, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=39, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(40)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=41, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=42, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(43)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=44, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)
    
    module = convolution(module_id=45, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)
    
    module = shortcut(46)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=47, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=48, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(49)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=50, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=51, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(52)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=53, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=54, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)

    module = shortcut(55)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=56, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=57, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(58)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=59, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=60, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = shortcut(61)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=62, in_channels=output_filters[-1], filters=1024, size=3, stride=2, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = convolution(module_id=63, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=64, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = shortcut(65)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=66, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=67, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = shortcut(68)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=69, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=70, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = shortcut(71)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=72, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=73, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = shortcut(74)
    module_list.append(module)
    output_filters.append(output_filters[1:][-3])

    module = convolution(module_id=75, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=76, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = convolution(module_id=77, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=78, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = convolution(module_id=79, in_channels=output_filters[-1], filters=512, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=80, in_channels=output_filters[-1], filters=1024, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(1024)

    module = convolution(module_id=81, in_channels=output_filters[-1], filters=255, size=1, stride=1, pad=1, batch_normalize=False, activation="linear")
    module_list.append(module)
    output_filters.append(255)

    module = yolo(82, mask=[6, 7, 8], anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)], num_classes=80)
    module_list.append(module)

    module = route(83)
    module_list.append(module)
    output_filters.append(sum([output_filters[1:][i] for i in [-4]]))

    module = convolution(module_id=84, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module  =  upsample(85, stride=2)
    module_list.append(module)

    module = route(87)
    module_list.append(module)
    output_filters.append(sum([output_filters[1:][i] for i in [-1, 61]]))


    module = convolution(module_id=87, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=88, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=89, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=90, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=91, in_channels=output_filters[-1], filters=256, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=92, in_channels=output_filters[-1], filters=512, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(512)

    module = convolution(module_id=93, in_channels=output_filters[-1], filters=255, size=1, stride=1, pad=1, batch_normalize=False, activation="linear")
    module_list.append(module)
    output_filters.append(255)

    module = yolo(94, mask=[3, 4, 5], anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)], num_classes=80)
    module_list.append(module)

    module = route(95)
    module_list.append(module)
    output_filters.append(sum([output_filters[1:][i] for i in [-4]]))

    module = convolution(module_id=96, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = upsample(97, stride=2)
    module_list.append(module)

    module = route(98)
    module_list.append(module)
    output_filters.append(sum([output_filters[1:][i] for i in [-1, 36]]))

    module = convolution(module_id=99, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=100, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=101, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=102, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=103, in_channels=output_filters[-1], filters=128, size=1, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(128)

    module = convolution(module_id=104, in_channels=output_filters[-1], filters=256, size=3, stride=1, pad=1, batch_normalize=True, activation="leaky")
    module_list.append(module)
    output_filters.append(256)

    module = convolution(module_id=105, in_channels=output_filters[-1], filters=255, size=1, stride=1, pad=1, batch_normalize=False, activation="linear")
    module_list.append(module)
    output_filters.append(255)

    module = yolo(106, mask=[0, 1, 2], anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)], num_classes=80)
    module_list.append(module)

    return hyperparams, module_list


















