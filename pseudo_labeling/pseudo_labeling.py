import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sympy import flatten
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
import os
import sys
from numpy.linalg import norm
import cv2
from glob import glob
from tqdm import tqdm
import time
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from transformers import AutoImageProcessor, AutoModel

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Already folder exists')

def roi(model_input, box_size, dataset):
    visualize = torch.permute(model_input[0], (1,2,0)).detach().cpu().numpy()
    tmp_img = visualize.copy()
    
    if dataset=='orfd':
        width_offset = 16

        flatten_indices = []
        for i in range(box_size):
            for j in range(box_size):
                left = j * grid_size
                upper = i * grid_size
                right = left + grid_size
                lower = upper + grid_size
                if (box_size-6<=i<=box_size-1)&(box_size-width_offset-1>=j>=width_offset):
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
    
    elif dataset=='gurka':
        top_width_offset = 3
        left_side_width_offset = 10
        right_side_width_offset = 12

        flatten_indices = []
        for i in range(box_size):
            for j in range(box_size):
                left = j * grid_size
                upper = i * grid_size
                right = left + grid_size
                lower = upper + grid_size
                if (box_size-23<=i<=box_size-22)&(box_size//2-top_width_offset<=j<=box_size//2+top_width_offset):
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-10<=i<=box_size-0)&(box_size//2-left_side_width_offset-1<=j<=box_size//2-left_side_width_offset):
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-10<=i<=box_size-0)&(box_size//2+right_side_width_offset<=j<=box_size//2+right_side_width_offset+1):
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
    
    return flatten_indices

def find_drivable_indices(box_size, tmp_map):
    flatten_indices = []
    for i in range(box_size):
        for j in range(box_size):
            left = j * grid_size
            upper = i * grid_size
            right = left + grid_size
            lower = upper + grid_size
            
            unique, counts = np.unique(tmp_map[upper:lower, left:right], return_counts=True)
            uniq_cnt_dict = dict(zip(unique, counts))
            if 1 in uniq_cnt_dict.keys():
                if (uniq_cnt_dict[1] / grid_size**2)>0.9:
                    flatten_indices.append(i*box_size+j)
    
    return flatten_indices

def crf(image, annot, resize_shape):
    colors, labels = np.unique(annot, return_inverse=True)
    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(annot.shape[1], annot.shape[0], 2)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=5, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=25, srgb=3, rgbim=np.array(image.resize(resize_shape)),
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run five inference steps.
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    drivable_map = np.argmax(Q, axis=0).reshape((resize_shape[1], resize_shape[0]))
    
    return drivable_map

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool_))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float32(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float32)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float32)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float32)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou

def fine_drivable(img, depth, model_output, flatten_indices, height, width, output_size, iter):
    mean = torch.mean(model_output[0][flatten_indices], dim=0)
    cosine_sim = F.cosine_similarity(model_output[0], mean.unsqueeze(0), dim=1)
    if depth is not None:
        if iter>0:
            # depth = depth.astype(np.float32)
            depth = torch.from_numpy(depth)
            depth = depth.permute(1,2,0)
            depth = torch.flatten(depth, start_dim=0, end_dim=1)
            model_output_ = torch.cat((model_output[0].detach().cpu(), depth), dim=1)#.to(device)
            mean = torch.mean(model_output_[flatten_indices], dim=0)
            cosine_sim = F.cosine_similarity(model_output_, mean.unsqueeze(0), dim=1)
    
    norm_cosine = cosine_sim / torch.max(cosine_sim)
    
    threshold_norm_cosine = norm_cosine.clone()
    threshold_norm_cosine[threshold_norm_cosine < threshold] = 0
    threshold_norm_cosine[threshold_norm_cosine >= threshold] = 1
    drivable_map = threshold_norm_cosine.reshape((output_size[0], output_size[1]))
    drivable_map_np = drivable_map.detach().cpu().numpy()
    resized = cv2.resize(drivable_map_np, (width, height), interpolation=cv2.INTER_NEAREST)

    crf_drivable_map = crf(img, resized, (width, height))
    
    return crf_drivable_map

def main():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', crop_size={'height':img_size, 'width':img_size}, size={'height':img_size, 'width':img_size})
    
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float64)
    for folder in folders:
        img_path = os.path.join(base_path, f'{folder}/image_data')
        depth_path = os.path.join(base_path, f'{folder}/dense_depth_npy')
        gt_path = os.path.join(base_path, f'{folder}/gt_image')
        
        save_path = os.path.join(base_path, f'{folder}/{save_folder_name}')
        makedirs(save_path)
        
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
        with torch.no_grad():
            for i in tqdm(img_list):
                img_name = i
                img = Image.open(os.path.join(img_path, f'{img_name}'))
                img_np = np.array(img)
                oriHeight, oriWidth, _ = img_np.shape
                
                # depth = cv2.imread(os.path.join(depth_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
                depth = np.load(os.path.join(depth_path, img_name.split('.')[0]+'.npy'))
                depth = cv2.resize(depth, (box_size, box_size), interpolation=cv2.INTER_NEAREST)
                depth_np = np.array(depth)
                depth_np = depth_np # (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
                depth_np = np.expand_dims(depth_np, axis=0)
                depth_np = None
                
                inputs = processor(images=img_np, return_tensors="pt")
                output_size = int(inputs.pixel_values[0].shape[1]/grid_size)
                
                model_input = inputs.pixel_values.to(device)
                model_output = dinov2_vitg14.get_intermediate_layers(model_input)[0]#.cpu().numpy()
                
                # min_vals = model_output.min(dim=-1, keepdim=True).values
                # max_vals = model_output.max(dim=-1, keepdim=True).values
                # scaled_tensor = (model_output - min_vals) / (max_vals - min_vals)
                # model_output = scaled_tensor

                for j in range(num_iter):
                    if j==0:
                        flatten_indices = roi(model_input, box_size, dataset)
                        crf_drivable_map = fine_drivable(img, depth_np, model_output, flatten_indices, img_size, img_size, (output_size, output_size), j)
                    else:
                        flatten_indices1 = find_drivable_indices(box_size, crf_drivable_map)
                        crf_drivable_map1 = fine_drivable(img, depth_np, model_output, flatten_indices1, oriHeight, oriWidth, (output_size, output_size), j)

                cv2.imwrite(filename=os.path.join(save_path, f'{img_name}'), img=(crf_drivable_map1*255))
                
                if dataset=='gurka':
                    continue
                
                label_img_name = img_name.split('.')[0]+"_fillcolor.png"
                label_dir = os.path.join(gt_path, label_img_name)
                label_image = cv2.cvtColor(cv2.imread(label_dir), cv2.COLOR_BGR2RGB)
                label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
                label[label_image[:,:,2] > 200] = 1
                
                conf_mat += confusion_matrix(np.int_(label), np.int_(crf_drivable_map1), num_labels)

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    print ('glob acc : {0:.3f}, pre : {1:.3f}, recall : {2:.3f}, F_score : {3:.3f}, IoU : {4:.3f}'.format(globalacc, pre, recall, F_score, iou))
    
if __name__ == "__main__":
    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dinov2_vitg14.eval().to(device)
    
    img_size = 644 # 644
    threshold = 0.55 # 0.5
    grid_size = 14
    box_size = img_size // grid_size  # num of grid per row and column
    num_labels = 2 # Drivable / Non-drivable
    num_iter = 2
    
    dataset = 'gurka' # orfd
    base_path = f'/media/imlab/HDD/{dataset}'
    folders = ['training', 'testing', 'validation']
    folders = ['0']
    
    save_folder_name = 'pseudo_labeling_3roi'
    
    main()