 # The train file for network
# Based on pytorch 1.8
# Use the docker: nlos_trans:1.8 in Ubuntu
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

from util.SpadDataset import SpadDataset
from util.SetRandomSeed import set_seed, worker_init
from util.SaveChkp import save_checkpoint
from util.MakeDataList import makelist
import util.SetDistTrain as utils
from tqdm import tqdm
from util.LFEDataset import LFEDataset,NLOSDataset
from sklearn.metrics import accuracy_score as ACC
 
from metric import RMSE, PSNR, SSIM ,AverageMeter, crop_to_cal,MAD
from tools import*
import torch.nn.functional as F
import cv2
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from models import model_tst_NLOS_NBlks_phy,embedfeature

def main():
    
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
    # int_path = '/data/yueli/nlos_sp_output/nips_cs/traditional_algos/syn_bike/CSA/pred_8/'
    # gt_path = '/data/yueli/nlos_sp_output/nips_cs/traditional_algos/syn_bike/CSA/gt_8/'
    mat_path = '/data2/yueli/code/NeTF_public-main/logs/syn_bike/mea_8/'
    gt_path = '/data2/yueli/code/NeTF_public-main/data/syn_bike/gt/'
    with torch.no_grad():
        for i in range(0,290,1):
            mat = scio.loadmat(mat_path + f'{i}_bike/model/predicted_volume49_8.mat')
            volume_rho = mat['volume_rho']
            volume = mat['volume']
            albedo = volume_rho * volume
            # print(volume_rho.shape,volume.shape,albedo.shape)
            int_re = np.max(albedo,axis=1)
            int_re = np.transpose(int_re,[1,0])
            int_re = int_re[::-1,::-1]
            pred_int = int_re/np.max(int_re)
            # print(int_re.shape)
            cv2.imwrite('/data2/yueli/code/nlos_cs_nips2023/bike_syn_netf/mea8/' + f'{i}_int.png', (int_re) /np.max(int_re)*255)

            gt = cv2.imread(gt_path+f'{i}int.png',-1)/255
            gt = cv2.resize(gt,(128,128))
            # print(pred_int.shape,gt.shape)
            # print(pred_int.mean(),gt.mean())

            int_re = torch.from_numpy(pred_int[None][None]).cuda().type(dtype)
            target = torch.from_numpy(gt[None][None]).cuda().type(dtype)

            foreground_gt = target.detach().cpu().numpy() * 255
            foreground_gt[foreground_gt < 10] = 0
            foreground_gt[foreground_gt > 0] = 1
            
            foreground_pred = int_re.detach().cpu().numpy() * 255
            foreground_pred[foreground_pred < 10] = 0
            foreground_pred[foreground_pred > 0] = 1       
            acc_sum = 0
            for i in range(int_re.shape[0]):
                # print(foreground_pred[i,0].shape,foreground_gt[i,0].shape)
                acc_sum+=ACC(foreground_pred[i,0], foreground_gt[i,0])
            acc_sum/=int_re.shape[0]
            
            intensity_metrics['rmse'].update(rmse(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['psnr'].update(psnr(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['ssim'].update(ssim(int_re.clamp(0,1), target).cpu().item())
            intensity_metrics['acc'].update(acc_sum)
            
            
         
    log_str = ''
    for k in metric_list:
            log_str += '{:s} {:.5f} | '.format(k, intensity_metrics[k].item())
    log_str +=   'depth mad {:.5f} | '.format(depth_metrics['mad'].item())    
    log_str +=   'depth madwomask {:.5f} | '.format(depth_metrics['madwomask'].item())    
    print(log_str)
    # print('average time is ',total_time/len(val_loader))


if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




