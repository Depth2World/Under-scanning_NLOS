
import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from datetime import datetime
import scipy.io as scio

from util.SetRandomSeed import set_seed, worker_init
from util.SaveChkp import save_checkpoint
from util.MakeDataList import makelist
import util.SetDistTrain as utils
from yueli.code.nlos_cs_nips2023.pro.Train_ import train
from tqdm import tqdm
import torch.nn.functional as F


from models.utils_pytorch import phasor_1_10
from models.utils_pytorch import fk_1_10
from models.utils_pytorch import lct_1_10


import cv2
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim

def main():
    
    # baseline   
    
    spatial = 128
    
    
    temp_bin = 512
    bin_len = ( 512 // temp_bin ) * 0.0096

    model = phasor_1_10.phasor(spatial=128, crop=temp_bin, bin_len=bin_len, sampling_coeff=2.0, cycles=5,dnum=1)
    # model = fk_1_10.lct_fk(spatial=128, crop=512, bin_len=bin_len,dnum=1)
    # model = lct_1_10.lct(spatial=128, crop=512, bin_len=bin_len,method='lct',dnum=1)
    
    model.cuda()
    model = torch.nn.DataParallel(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    print("Start eval...")
    rw_path  = '/data/yueli/dataset/align_fk_256_512/'
    rw_path = '/data/yueli/dataset/align_fk_256_512_meas_10min'
    
    out_path = f'/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_meas_10min/compressed_tradition_out_resize_before/sptial{spatial}/rsd/'
    
    # out_path = f'/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_data/interp128/sptial{spatial}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    ims = []
    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])['final_meas']  #sig final_meas measlr
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 64, 64,2048  8ps
        # subsampled to 128 128
        M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        # spatial sample 
        ds = 128 // spatial
        M_wnoise = M_wnoise[:,ds//2::ds,ds//2::ds,:]  
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        M_mea = torch.from_numpy(M_wnoise[None])  
        
         # resize before
        M_mea = F.interpolate(M_mea,[512,128,128])
        # resized_mea = M_mea.detach().cpu().numpy()[0,0].transpose([1,2,0])
        # scio.savemat(out_path  + files[i][:-4] + f'{spatial}to128.mat',{'resized_mea':resized_mea})
        
        print(M_mea.size())
        with torch.no_grad():
            model.eval()
            re = model(M_mea,[0,0,0],[temp_bin,temp_bin,temp_bin])
            
            volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
            # zdim = volumn_MxNxN.shape[0] * 100 // 128
            # volumn_MxNxN = volumn_MxNxN[:zdim]
            # print('volumn min, %f' % volumn_MxNxN.min())
            # print('volumn max, %f' % volumn_MxNxN.max())
            
            volumn_MxNxN[volumn_MxNxN < 0] = 0
            
            front_view = np.max(volumn_MxNxN, axis=0)
            front_view = front_view / np.max(front_view)
            front_view = (front_view*255).astype(np.uint8)
            # front_view = cv2.resize(front_view, (128, 128))
            depth_view = np.argmax(volumn_MxNxN, axis=0)
            depth_view = depth_view.astype(np.float32)/np.max(depth_view)
            depth_view = (depth_view*255).astype(np.uint8)
            # depth_view = cv2.resize(depth_view, (128, 128))
            
            # cv2.imwrite(out_path + files[i] + f'_dep_pred_temp{temp_bin}.png',depth_view)
            cv2.imwrite(out_path + files[i] + f'_int_pred_temp{temp_bin}.png',front_view)
            
            # dep_np = dep_re.squeeze(0).data.cpu().numpy()
            # dep_np = (dep_np).clip(0,1)
            # dep_np = dep_np/np.max(dep_np)
            # cv2.imwrite(out_path + files[i] + f'_dep_pred_temp{temp_bin}.png',dep_np.squeeze(0)*255)
            
            # int_re = int_re.squeeze(0).data.cpu().numpy()
            # int_re = (int_re).clip(0,1)
            # int_re = int_re/np.max(int_re)
            # cv2.imwrite(out_path + files[i] + f'_int_pred_temp{temp_bin}.png',int_re.squeeze(0)*255)
            
            # import matplotlib.pyplot as plt
            # plt.imshow(dep_np.squeeze())
            # plt.savefig(out_path + files[i] + '_pred_512.png')
            # scio.savemat(out_path + files[i] + '_pred_512.mat',{'dep_pre':dep_np})

    
    
    
if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




