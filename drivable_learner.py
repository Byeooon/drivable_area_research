import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data_load import get_train_dataloaders
from visualize import visualize_data
from nets import DrivableNet
from loss import SegmantationLoss

os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
if device=="cuda": torch.cuda.empty_cache()
print('학습을 진행하는 기기:',device)

class DrivableLearner():
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(self.timestamp))
        
    def train(self):
        model = torch.load('/home/julio981007/drivable_area_research/checkpoints/dinov2_seg_head/dinov2_vitg14_ade20k_linear_head.pth')
        print("Model's state_dict:")
        print(model['state_dict'])
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        sys.exit()
        
        model = DrivableNet(device=device)
        
        train_loader, val_loader = get_train_dataloaders('/media/imlab/HDD/ORFD/', batch_size=self.args.batch_size)
        
        optimizer_RoadSeg = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        # criterionSegmentation = SegmantationLoss(class_weights=None).to(device)
        criterion = torch.nn.BCELoss().to(device)
        best_vloss = 1000000.
        for epoch in tqdm(range(self.args.num_epochs)):
            
            train_running_loss = 0.0
            for i, (img, depth, pcd, gt) in enumerate(tqdm(train_loader)):
                # print("Images shape:", img.shape)
                # print("Depth shape:", depth.shape)
                # print("LiDAR data shape:", pcd.shape)
                # print("Ground truth images shape:", gt.shape)
                
                output = model(img.to(device), depth.to(device))
                output = F.interpolate(output, size=gt.shape[2:], mode='bicubic', align_corners=False)
                
                optimizer_RoadSeg.zero_grad()
                loss_segmentation = criterion(output, gt.to(device))
                loss_segmentation.backward()
                optimizer_RoadSeg.step()
                
                train_running_loss += loss_segmentation.item()
                
            with torch.no_grad():
                val_running_loss = 0.0
                for j, (img, depth, pcd, gt) in enumerate(tqdm(val_loader)):
                    output = model(img.to(device), depth.to(device))
                    output = F.interpolate(output, size=gt.shape[2:], mode='bicubic', align_corners=False)
                    
                    loss_segmentation = criterion(output, gt.to(device))
                    val_running_loss += loss_segmentation.item()
            
            train_cost = train_running_loss / len(train_loader)
            val_cost = val_running_loss / len(val_loader)
            print('[%d] train loss: %.3f / val loss: %.3f' %(epoch + 1, train_cost, val_cost))
            
            self.writer.flush()
            if val_cost < best_vloss:
                best_vloss = val_cost
                model_path = 'model_{}_{}'.format(self.timestamp, epoch)
                torch.save(model.state_dict(), model_path)
        
        print('Finished Training')
        
if __name__ == "__main__":
    pass