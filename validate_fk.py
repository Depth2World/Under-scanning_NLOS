### Modify
# Line 41 trained_model path
# Line 57 cvpr2023 data real-world path 
# Line 58 ouptut path 


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
 
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim
from models import embedfeature
import cv2

def main():
    
    # baseline  
    ds = 8
    spatial = 128//ds
    model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.0096*2,views=1,wall_size=2, sp_ds_scale=ds)
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
    print("Start eval...")
    rw_path  = '/data2/yueli/dataset/NLOS_RW/align_fk_256_512'
    out_path = 'xxx'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    for i in range(len(all_file)): 

        transient_data = scio.loadmat(all_file[i])
        transient_data = transient_data['final_meas'] 
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   
        M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        # spatial sample 
        
        if True:
            raw = torch.from_numpy(M_wnoise.astype(np.float32))              # (1/3, t, h, w)
            mask = torch.zeros_like(raw) # 1 t h w
            for index_i in range(ds//2, 128, ds):
                for index_j in range(ds//2, 128, ds):
                    mask[:,:,index_i,index_j] = 1
            M_mea =  raw * mask
            M_mea = M_mea.unsqueeze(0)
        else:
            M_wnoise = M_wnoise[:,:,ds//2::ds,ds//2::ds]  
            M_mea = torch.from_numpy(M_wnoise[None])  
            M_mea = F.interpolate(M_mea,[512,128,128])
            
        print(M_mea.shape)
        with torch.no_grad():
            model.eval()
            # pred_mea, vlo, im_re = model(M_mea)
            vlo, im_re = model(M_mea)
            front_view = im_re.detach().cpu().numpy()[0, 0]
            vlo = vlo.detach().cpu().numpy()[0]
            # resized_mea = pred_mea.detach().cpu().numpy()[0,0].transpose([1,2,0])
            scio.savemat(out_path  + files[i][:-4] + f'{spatial}to128_vlo.mat',{'vlo':vlo})
            # scio.savemat(out_path  + files[i][:-4] + f'{spatial}to128_mea.mat',{'resized_mea':resized_mea})
            cv2.imwrite(out_path + files[i][:-4] + f'{spatial}to128.png', (front_view / np.max(front_view))*255)
if __name__=="__main__":
    main()




