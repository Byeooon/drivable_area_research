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

class DrivableLearner():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('{}/trainer_{}'.format(self.args.ckpt_dir, self.timestamp))
        
    def train(self):
        model = DrivableNet(self.args.depth, device=self.device)
        
        train_loader, val_loader = get_train_dataloaders(self.args.dataset_dir, img_height=self.args.img_height, img_width=self.args.img_width, 
                                                         batch_size=self.args.batch_size)
        
        optimizer_RoadSeg = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        # criterionSegmentation = SegmantationLoss(class_weights=None).to(self.device)
        criterion = torch.nn.BCELoss().to(self.device)
        best_vloss = 1000000.
        for epoch in tqdm(range(1, self.args.num_epochs+1)):
            
            train_running_loss = 0.0
            for i, (img, depth, pcd, gt) in enumerate(tqdm(train_loader)):
                # print("Images shape:", img.shape)
                # print("Depth shape:", depth.shape)
                # print("LiDAR data shape:", pcd.shape)
                # print("Ground truth images shape:", gt.shape)
                
                img = img.to(self.device)
                if self.args.depth:
                    depth = depth.to(self.device)
                else:
                    pass
                
                output = model(img, depth)
                output = F.interpolate(output, size=gt.shape[2:], mode='nearest', align_corners=False)
                
                optimizer_RoadSeg.zero_grad()
                loss_segmentation = criterion(output, gt.to(self.device))
                loss_segmentation.backward()
                optimizer_RoadSeg.step()
                
                train_running_loss += loss_segmentation.item()
            
            with torch.no_grad():
                val_running_loss = 0.0
                for j, (img, depth, pcd, gt) in enumerate(tqdm(val_loader)):
                    output = model(img.to(self.device), depth.to(self.device))
                    output = F.interpolate(output, size=gt.shape[2:], mode='nearest', align_corners=False)
                    
                    loss_segmentation = criterion(output, gt.to(self.device))
                    val_running_loss += loss_segmentation.item()
            
            train_cost = train_running_loss / len(train_loader)
            self.writer.add_scalar("Loss/train", train_cost, epoch)
            val_cost = val_running_loss / len(val_loader)
            self.writer.add_scalar("Loss/val", val_cost, epoch)
            print('[%d] train loss: %.3f / val loss: %.3f' %(epoch, train_cost, val_cost))
            
            self.writer.flush()
            if val_cost < best_vloss:
                best_vloss = val_cost
                model_path = os.path.join(self.args.ckpt_dir, 'model_{}_{}'.format(self.timestamp, epoch))
                torch.save(model.state_dict(), model_path)
                print(f'Model saved at > {model_path}')
        
        self.writer.close()
        print('Finished Training')
        
if __name__ == "__main__":
    pass