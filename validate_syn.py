### Modify
# Line 41 trained_model path
# Line 60 synthetic data path 
# Line 77 ouptut path 


import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import scipy.io as scio
from util.SetRandomSeed import set_seed, worker_init
from util.SaveChkp import save_checkpoint
import util.SetDistTrain as utils
from tqdm import tqdm
from util.LFEDataset import LFEDataset,NLOSDataset
from sklearn.metrics import accuracy_score as ACC
from metric import RMSE, PSNR, SSIM ,AverageMeter, crop_to_cal,MAD
from tools import*
import torch.nn.functional as F
from pro.Loss import criterion_KL, criterion_L2
from models import embedfeature

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)

def main():
    
    ds_scale = 8
    model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.02,views=1,wall_size=2, sp_ds_scale=ds_scale)
    model_path = 'xxx.pth'
    
    model.cuda()
    model = torch.nn.DataParallel(model)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model_dict = model.state_dict()
    ckpt_dict = checkpoint['state_dict']
    model_dict.update(ckpt_dict)
    #for k in ckpt_dict.keys():
    #    model_dict.update({k[7:]: ckpt_dict[k]})
    model.load_state_dict(model_dict)
    print("Loaded and update model states!")
    print("Start eval...")

    folder_path = ['/data2/yueli/dataset/LFE_dataset/bike']
    shineness = [0]
    val_data = LFEDataset(root=folder_path, # dataset root directory
                                shineness=shineness,
                                for_train=False,
                                ds=1,               # temporal down-sampling factor
                                clip=512,           # time range of histograms
                                size=256,           # measurement size (unit: px)
                                scale=1,            # scaling factor (float or float tuple)
                                background=[0.05,2],# background noise rate (float or float tuple)
                                target_size=128,    # target image size (unit: px)
                                target_noise=0.01,  # standard deviation of target image noise
                                color='gray',       # color channel(s) of target image
                                sp_ds=ds_scale, # spatial resolution downsample
                                mask=True)         # mea * mask or not 
    
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    out_path = 'xxx/syn_bike/'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path,'int'), exist_ok=True)
        # os.makedirs(os.path.join(out_path,'dep'), exist_ok=True)
        
    rmse = RMSE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()
    mad = MAD().cuda()
    
    metric_list = ['rmse', 'psnr', 'ssim','acc']
    intensity_metrics = {k: AverageMeter() for k in metric_list}
    depth_metrics = {'mad': AverageMeter(),'madwomask': AverageMeter() }
    niter = 0
    total_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        model.eval()
        for sample in tqdm(val_loader):
            # if niter==10:break;
            M_mea, raw_mea, dep_gt, img_gt= sample["ds_meas"].type(dtype), sample["raw_meas"].type(dtype), sample["dep_gt"].type(dtype), sample["img_gt"].type(dtype)
            # print(M_mea.shape, raw_mea.shape)
            ###### predict ######
            starter.record()
            up_M_mea, re_vlo, inten_re, target, depth_re, targetd = model(M_mea, img_gt, dep_gt)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 计算时间
            total_time += curr_time
            
            int_re, depth_re, target ,targetd = inten_re.squeeze(1), depth_re.squeeze(1), target.squeeze(1), targetd.squeeze(1)
            front_view = int_re[0,...].cpu().numpy().transpose(1,2,0)
            depth_view = depth_re[0,...].cpu().numpy().transpose(1,2,0)
            view_gt = target[0].cpu().numpy().transpose(1,2,0)
            cv2.imwrite(os.path.join(out_path,'int/') + f'{niter}_int.png', (front_view/np.max(front_view)*255))
            # cv2.imwrite(os.path.join(out_path,'dep/') + f'{niter}_dep.png', (depth_view/np.max(depth_view)*255))
            
            foreground_gt = target.detach().cpu().numpy() * 255
            foreground_gt[foreground_gt < 10] = 0
            foreground_gt[foreground_gt > 0] = 1
            
            foreground_pred = int_re.detach().cpu().numpy() * 255
            foreground_pred[foreground_pred < 10] = 0
            foreground_pred[foreground_pred > 0] = 1       
            acc_sum = 0
            for i in range(int_re.shape[0]):
                acc_sum+=ACC(foreground_pred[i,0], foreground_gt[i,0])
            acc_sum/=int_re.shape[0]
            
            intensity_metrics['rmse'].update(rmse(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['psnr'].update(psnr(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['ssim'].update(ssim(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['acc'].update(acc_sum)
            niter += 1
         
    log_str = ''
    for k in metric_list:
            log_str += '{:s} {:.5f} | '.format(k, intensity_metrics[k].item())
    log_str +=   'depth mad {:.5f} | '.format(depth_metrics['mad'].item())    
    log_str +=   'depth madwomask {:.5f} | '.format(depth_metrics['madwomask'].item())    
    print(log_str)
    print('average time is ',total_time/len(val_loader))


if __name__=="__main__":
    main()




