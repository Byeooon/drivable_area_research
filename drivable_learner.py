import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

from data_load import get_train_dataloaders
from visualize import visualize_data
from nets import DrivableNet
from loss import SegmantationLoss

class DrivableLearner():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('{}'.format(self.args.ckpt_dir))
        
    def train(self):
        train_loader, val_loader = get_train_dataloaders(self.args.dataset_dir, img_height=self.args.img_height, img_width=self.args.img_width, 
                                                         batch_size=self.args.batch_size)
        
        num_patch = self.args.img_height // 14
        model = DrivableNet(self.args.depth, num_patch, device=self.device)
        
        optimizer_RoadSeg = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer_RoadSeg,
        #                                 lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                 last_epoch=-1,
        #                                 verbose=False)
        
        # criterionSegmentation = SegmantationLoss(class_weights=None).to(self.device)
        criterion = torch.nn.BCELoss().to(self.device)
        
        best_vloss = 1000000.
        patience_cnt=0
        
        for epoch in tqdm(range(1, self.args.num_epochs+1)):
            train_running_loss = 0.0
            for i, (img, depth, pcd, gt) in enumerate(tqdm(train_loader)):
                img = img.to(self.device)
                if self.args.depth:
                    depth = depth.to(self.device)
                else:
                    pass
                
                output = model(img, depth)
                output = F.interpolate(output, size=gt.shape[2:], mode='nearest')# , align_corners=True)
                
                optimizer_RoadSeg.zero_grad()
                loss_segmentation = criterion(output, gt.to(self.device))
                loss_segmentation.backward()
                optimizer_RoadSeg.step()
                
                train_running_loss += loss_segmentation.item()
                
                if i%self.args.summary_freq == 0:
                    self.writer.add_scalar("Step Loss/train", loss_segmentation.item(), i+1)
                    # for batch in range(self.args.batch_size):
                    self.writer.add_image(f'Image/input', img[0], i+1)
                    self.writer.add_image(f'Image/output', output[0], i+1)
            # scheduler.step()
            
            with torch.no_grad():
                val_running_loss = 0.0
                for j, (img, depth, pcd, gt) in enumerate(tqdm(val_loader)):
                    img = img.to(self.device)
                    if self.args.depth:
                        depth = depth.to(self.device)
                    else:
                        pass
                    
                    output = model(img, depth)
                    output = F.interpolate(output, size=gt.shape[2:], mode='nearest')# , align_corners=True)
                    
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
            else:
                patience_cnt += 1
                print(f'Patience Count : {patience_cnt}\n')
                if patience_cnt >= self.args.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        self.writer.close()
        print('Finished Training')
        
if __name__ == "__main__":
    pass