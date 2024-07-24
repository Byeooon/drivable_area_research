import sys
from tracemalloc import start
import torch
from torch import nn
import torch.nn.functional as F

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        
        for i, (name, param) in enumerate(self.dinov2_vitg14.named_parameters()):
            param.requires_grad = False
        
    def forward(self, img):
        x = self.dinov2_vitg14.get_intermediate_layers(img)[0]
        output = x
        
        return output

class DepthEncoder(nn.Module):
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=14, stride=14)
        self.norm1 = nn.LayerNorm([32, 16, 16])
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([64, 16, 16])
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([128, 16, 16])
        
        self.conv4 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.norm4 = nn.LayerNorm([output_channels, 16, 16])
        
        self.gelu = nn.GELU()
        
    def forward(self, depth):
        x = self.conv1(depth)
        x = self.gelu(self.norm1(x))
        
        x = self.conv2(x)
        x = self.gelu(self.norm2(x))
        
        x = self.conv3(x)
        x = self.gelu(self.norm3(x))
        
        x = self.conv4(x)
        x = self.gelu(self.norm4(x))
        
        output = x
        
        return output
    
class SegHead(nn.Module):
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([128, 16, 16])
        
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([64, 16, 16])
        
        self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([32, 16, 16])
        
        self.convt4 = nn.ConvTranspose2d(32, output_channels, kernel_size=14, stride=14)
        
        self.gelu = nn.GELU()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.convt1(x)
        x = self.gelu(self.norm1(x))
        
        x = self.convt2(x)
        x = self.gelu(self.norm2(x))
        
        x = self.convt3(x)
        x = self.gelu(self.norm3(x))
        
        x = self.convt4(x)
        output = self.sigmoid(x)
        
        return output
        
class DrivableNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.img_model = ImgEncoder().to(device=device)
        self.depth_model = DepthEncoder(in_channels=1, output_channels=256).to(device=device)
        self.seg_head = SegHead(in_channels=1792, output_channels=1).to(device=device)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        
    def forward(self, img, depth):
        x = self.img_model(img)
        y = self.depth_model(depth)
        y = self.flatten(y)
        
        concat = torch.concat((x, y), dim=-1)
        concat = concat.permute(0, 2, 1)
        concat = concat.view(img.shape[0], -1, 16, 16)
        
        out = self.seg_head(concat)
        
        return out