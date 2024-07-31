import os
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

def make_rgb_transform(smaller_edge_size: tuple) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
def make_depth_transform(smaller_edge_size: tuple) -> transforms.Compose:
    # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.NEAREST
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def bin_to_numpy(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:,:4]  #jk hesai40 data
    points[:, 3] = 1.0  # homogeneous
    return points

def custom_collate_fn(batch):
    img, depth, pcd, gt = zip(*batch)
    
    # 이미지, 깊이 맵, 그라운드 트루스는 기본 collate 함수 사용
    img = torch.utils.data.dataloader.default_collate(img)
    
    if None in depth:
        pass
    else:
        depth = torch.utils.data.dataloader.default_collate(depth)
        pcd = list(pcd)
    
    gt = torch.utils.data.dataloader.default_collate(gt)
    
    # pcd는 리스트로 유지
    return img, depth, pcd, gt

class DrivableAreaDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, depth=False):
        print(f"Dataset initialized for {split}")
        self.image_dir = os.path.join(base_dir, split, 'image_data')
        
        self.depth = depth
        if self.depth:
            self.depth_dir = os.path.join(base_dir, split, 'dense_depth')
            self.lidar_dir = os.path.join(base_dir, split, 'lidar_data')
            print(f"Depth directory: {self.depth_dir}")
            print(f"LiDAR directory: {self.lidar_dir}")
        
        self.gt_dir = os.path.join(base_dir, split, 'gt_image')
        self.rgb_transform = transform[0]
        self.depth_transform = transform[1]
        self.samples = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        print(f"Image directory: {self.image_dir}")
        print(f"Ground Truth directory: {self.gt_dir}")
        print(f"Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name.replace('.png', '_fillcolor.png'))

        # 이미지 로드
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
            
        depth = None
        lidar_tensor = None
        if self.depth:
            depth_path = os.path.join(self.depth_dir, img_name)
            lidar_path = os.path.join(self.lidar_dir, img_name.replace('.png', '.bin'))
            
            # Depth 로드
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth = np.array(depth)
            # depth_tensor = transforms.ToTensor()(depth)
            if self.depth_transform:
                depth = self.depth_transform(depth)
            if not isinstance(depth, torch.Tensor):
                depth = transforms.ToTensor()(depth)

            # LiDAR 데이터 로드
            lidar_data = bin_to_numpy(lidar_path)
            print('lidar : ', lidar_data.shape)
            lidar_tensor = torch.from_numpy(lidar_data).float()

        # Ground Truth 이미지 로드
        gt_image = cv2.imread(gt_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        gt_image = np.array(gt_image)
        gt_image[gt_image<255] = 0
        gt_tensor = transforms.ToTensor()(gt_image)

        return image, depth, lidar_tensor, gt_tensor

def get_train_dataloaders(base_dir, img_height, img_width, batch_size=1, num_workers=0):
    rgb_transform = make_rgb_transform(smaller_edge_size=(img_height, img_width))
    depth_transform = make_depth_transform(smaller_edge_size=(img_height, img_width))
    train_dataset = DrivableAreaDataset(base_dir, 'training', transform=(rgb_transform, depth_transform), depth=False)
    val_dataset = DrivableAreaDataset(base_dir, 'validation', transform=(rgb_transform, depth_transform), depth=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def get_test_dataloaders(base_dir, img_height, img_width, batch_size=1, num_workers=0):
    rgb_transform = make_rgb_transform(smaller_edge_size=(img_height, img_width))
    depth_transform = make_depth_transform(smaller_edge_size=(img_height, img_width))
    test_dataset = DrivableAreaDataset(base_dir, 'testing', transform=(rgb_transform, depth_transform), depth=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    return test_loader