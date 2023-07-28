import sys
import os
sys.path.append('C:/CNN/AVA_DF2')

import torch
import torch.nn as nn
import numpy as np
import timm
import yaml
from utils.general import ConfigObject

from model.resnext import resnext101
from model.BiFPN import BiFPN
from model.cfam import CFAMBlock

# Detection Head
class Detect(nn.Module):
    
    def __init__(self, no=80, anchors=(), ch=(), training=True):  # detection layer
        super(Detect, self).__init__()
        self.training = training
        self.no = no  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)

# Multi-Task Action Fusion 3D (Model)
class MTA_F3D_MODEL(nn.Module):
    def __init__(self, cfg):
        super(MTA_F3D_MODEL, self).__init__()
        self.cfg = cfg
        self.num_frames = cfg.DATA.NUM_FRAMES
        
        # 2D Backbone
        self.backbone_2d = timm.create_model('cspresnext50', features_only=True, pretrained=True)
        num_ch_2d = [64, 256, 512, 1024, 2048]
        
        # 3D backbone
        self.backbone_3d = resnext101()
        num_ch_3d = [64, 256, 512, 1024, 2048]
        
        if cfg.WEIGHTS.BACKBONE_3D:
            self.backbone_3d = self.backbone_3d.cuda()
            self.backbone_3d = nn.DataParallel(self.backbone_3d, device_ids=None) # Because the pretrained backbone models are saved in Dataparalled mode
            pretrained_3d_backbone = torch.load(cfg.WEIGHTS.BACKBONE_3D)
            backbone_3d_dict = self.backbone_3d.state_dict()
            pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if k in backbone_3d_dict} # 1. filter out unnecessary keys
            backbone_3d_dict.update(pretrained_3d_backbone_dict) # 2. overwrite entries in the existing state dict
            self.backbone_3d.load_state_dict(backbone_3d_dict) # 3. load the new state dict
            self.backbone_3d = self.backbone_3d.module # remove the dataparallel wrapper

        # Neck 
        BiFPN_fsize = 64
        self.BiFPN_2d = BiFPN(num_ch_2d, BiFPN_fsize)
        
        # Neck (2D + 3D)
        time_p3 = max(int(self.num_frames / 4), 1)
        self.conv3D_p3 = nn.Conv3d(num_ch_3d[2], num_ch_3d[2], kernel_size=(time_p3, 1, 1), stride=(time_p3, 1, 1), padding=(0, 0, 0))
        time_p4 = max(int(self.num_frames / 8), 1)
        self.conv3D_p4 = nn.Conv3d(num_ch_3d[3], num_ch_3d[3], kernel_size=(time_p4, 1, 1), stride=(time_p4, 1, 1), padding=(0, 0, 0))
        time_p5 = max(int(self.num_frames / 16), 1)
        self.conv3D_p5 = nn.Conv3d(num_ch_3d[4], num_ch_3d[4], kernel_size=(time_p5, 1, 1), stride=(time_p5, 1, 1), padding=(0, 0, 0))
        
        self.cfam_p3 = CFAMBlock(BiFPN_fsize+num_ch_3d[2], 256)
        self.cfam_p4 = CFAMBlock(BiFPN_fsize+num_ch_3d[3], 512)
        self.cfam_p5 = CFAMBlock(BiFPN_fsize+num_ch_3d[4], 1024)
        
        # Head
        self.head_bbox = Detect(no = 4+1,
                                anchors = cfg.MODEL.ANCHORS,
                                ch = [BiFPN_fsize]*len(cfg.MODEL.ANCHORS),
                                training = False)
        self.head_clo = Detect(no = cfg.nc,
                                anchors = cfg.MODEL.ANCHORS,
                                ch = [BiFPN_fsize]*len(cfg.MODEL.ANCHORS),
                                training = False)
        self.head_act = Detect(no = cfg.MODEL.NUM_CLASSES,
                                anchors = cfg.MODEL.ANCHORS,
                                ch = [256,512,1024],
                                training = False)
        
    def forward(self, x):
        
        # input
        x_3d = x # input clip
        x_2d = x[:, :, -1, :, :] # Last frame of the clip that is read
        
        # Backbone
        fs_2d = self.backbone_2d(x_2d)
        _, fs_3d = self.backbone_3d(x_3d)
        
        # Neck
        fs_2d = self.BiFPN_2d(fs_2d)
        x_2d_p3 = fs_2d[2]
        x_2d_p4 = fs_2d[3]
        x_2d_p5 = fs_2d[4]
        
        x_3d_p3 = self.conv3D_p3(fs_3d[2])
        x_3d_p3 = torch.squeeze(x_3d_p3, dim=2)
        x_3d_p4 = self.conv3D_p4(fs_3d[3])
        x_3d_p4 = torch.squeeze(x_3d_p4, dim=2)
        x_3d_p5 = self.conv3D_p5(fs_3d[4])
        x_3d_p5 = torch.squeeze(x_3d_p5, dim=2)
        
        x_p3 = torch.cat((x_3d_p3, x_2d_p3), dim=1)
        x_p3 = self.cfam_p3(x_p3)
        x_p4 = torch.cat((x_3d_p4, x_2d_p4), dim=1)
        x_p4 = self.cfam_p4(x_p4)
        x_p5 = torch.cat((x_3d_p5, x_2d_p5), dim=1)
        x_p5 = self.cfam_p5(x_p5)
        
        # Head
        out_bboxs = self.head_bbox([x_2d_p3, x_2d_p4, x_2d_p5])
        out_clos = self.head_clo([x_2d_p3, x_2d_p4, x_2d_p5])
        out_acts = self.head_act([x_p3, x_p4, x_p5])
        
        return out_bboxs, out_clos, out_acts

if __name__ == '__main__':
    
    with open('cfg/deepfashion2.yaml', 'r') as f:
        _dict_df2 = yaml.safe_load(f)
        opt_df2 = ConfigObject(_dict_df2)
    
    with open('cfg/ava.yaml', 'r') as f:
        _dict_ava = yaml.safe_load(f)
        opt_ava = ConfigObject(_dict_ava)
        
    with open('cfg/model.yaml', 'r') as f:
        _dict_model = yaml.safe_load(f)
        opt_model = ConfigObject(_dict_model)
    
    with open('cfg/hyp.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    opt = ConfigObject({})
    opt.merge(opt_df2)
    opt.merge(opt_ava)
    opt.merge(opt_model)
        
    v = np.random.uniform(low=0.0, high=1.0, size= (opt.batch_size,3, opt.DATA.NUM_FRAMES, opt.img_size[0], opt.img_size[0]))
    v = torch.Tensor(v).cuda()
    
    model = MTA_F3D_MODEL(cfg = opt ).cuda()
    out_bboxs, out_clos, out_acts = model(v)
    print('bbox shape info')
    for i in out_bboxs:
        print(i.shape)
    print('clo shape info')
    for i in out_clos:
        print(i.shape)
    print('act shape info')
    for i in out_acts:
        print(i.shape)
