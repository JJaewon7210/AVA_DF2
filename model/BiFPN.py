import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

# Suppress all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # Define learnable weights
        self.w1= nn.Parameter(torch.ones(1))
        self.w2= nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.ones(1))
        
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        p7_td = p7_x
        p6_td = self.p6_td(p6_x) + F.interpolate(p7_td, scale_factor=2, mode='nearest')
        p5_td = self.p5_td(p5_x) + F.interpolate(p6_td, scale_factor=2, mode='nearest')
        p4_td = self.p4_td(p4_x) + F.interpolate(p5_td, scale_factor=2, mode='nearest')
        p3_td = self.p3_td(p3_x) + F.interpolate(p4_td, scale_factor=2, mode='nearest')
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(p4_x + p4_td + F.interpolate(p3_out, scale_factor=0.5, mode='nearest'))
        p5_out = self.p5_out(p5_x + p5_td + F.interpolate(p4_out, scale_factor=0.5, mode='nearest'))
        p6_out = self.p6_out(p6_x + p6_td + F.interpolate(p5_out, scale_factor=0.5, mode='nearest'))
        p7_out = self.p7_out(p7_x + p7_td + F.interpolate(p6_out, scale_factor=0.5, mode='nearest'))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p1 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p2 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[3], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[4], feature_size, kernel_size=1, stride=1, padding=0)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, inputs):
        c1, c2, c3, c4, c5 = inputs
        
        # Calculate the input column of BiFPN
        p1_x = self.p1(c1)        
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)        
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        
        features = [p1_x, p2_x, p3_x, p4_x, p5_x]
        return self.bifpn(features)