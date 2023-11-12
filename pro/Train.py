# The train function
import sys
sys.path.append("../util/")
import numpy as np 
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import test_on_align_cvpr, validate,test_on_align_fk,test_on_align_fk_10min
from util.SaveChkp import save_checkpoint
import util.SetDistTrain as utils
from pro.Loss import criterion_KL, criterion_L2, criterion_TV, criterion_NONZINDEX_MEAN, NLM,criterion_L2_var
#import cv2
import time
import torch.nn.functional as F

cudnn.benchmark = True
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
dtype = torch.cuda.FloatTensor
items = ["ALL", "KL", "L2intensity", "nlm", "spa", "TV","L2depth"]

def tv(x):
    return torch.sum(torch.abs(x[:,:, :, :, :-1] - x[:,:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:,:, :, :-1, :] - x[:,:, :, 1:, :]))

def tv_3d(x):
    return torch.sum(torch.abs(x[ :, :, :, :, :-1] - x[ :, :, :, :, 1:])) + \
           torch.sum(torch.abs(x[ :, :, :, :-1, :] - x[ :, :, :, 1:, :])) + \
           torch.sum(torch.abs(x[ :, :, :-1, :, :] - x[ :, :, 1:, :, :])) 
def sparisty(x):
    return torch.sum(torch.abs(x))

def uncertainty_loss(rot_view_inv_vars):
    return  torch.sum(torch.abs(rot_view_inv_vars[1,...] - rot_view_inv_vars[0,...])) + \
            torch.sum(torch.abs(rot_view_inv_vars[2,...] - rot_view_inv_vars[0,...])) + \
            torch.sum(torch.abs(rot_view_inv_vars[3,...] - rot_view_inv_vars[0,...]))

def train(model, train_loader, val_loader, optimer, epoch, n_iter,
            train_loss, val_loss, params, logWriter,logging):
    total_cost = 0
    for sample in tqdm(train_loader):
        # configure model state
        model.train()
        # load data and train the network
        M_mea, raw_mea, dep_gt, img_gt= sample["ds_meas"].type(dtype), sample["raw_meas"].type(dtype), sample["dep_gt"].type(dtype), sample["img_gt"].type(dtype)
        # print(M_mea.shape,raw_mea.shape)
        up_M_mea, re_vlo, inten_re, target, dep_re, targetd = model(M_mea, img_gt, dep_gt)
        # up_M_mea, re_vlo, inten_re, target, log_var = model(M_mea, img_gt, dep_gt)
        
        ## MUL
        # loss_kl = criterion_KL(up_M_mea, raw_mea)
        loss_kl = criterion_L2(up_M_mea, raw_mea)
        loss_l2 = criterion_L2(inten_re, target)
        loss_l2d = criterion_L2(dep_re, targetd)
        # loss_l2 = criterion_L2_var(inten_re, target, log_var)
        
        _,c,_,_,_ = re_vlo.shape
        loss_nlm = 0
        if c > 1 :
            for ch in range(c):
                vlo = re_vlo[:,ch,...]
                loss_nlm += NLM(vlo)
        else:
            loss_nlm = NLM(re_vlo)
        loss_tv3d = tv_3d(re_vlo)
        #possion = F.poisson_nll_loss(up_M_mea.flatten(), raw_mea.flatten(), log_input=False, reduction='mean')
        
        if params.sp_ds_scale>=1:
            loss =  params.kl_scale * loss_kl + 0 * loss_l2d + loss_l2  + params.nlm_scale * loss_nlm + params.tv3d_scale * loss_tv3d #+ params.po_scale * possion
        else: 
            loss = loss_l2
        
        
        # loss_sparisty  = sparisty(vlo)
    
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        n_iter += 1
        if utils.is_main_process():
            logWriter.add_scalar("loss_train/all", loss, n_iter)
            logWriter.add_scalar("loss_train/kl", loss_kl, n_iter)
            logWriter.add_scalar("loss_train/l2", loss_l2, n_iter)
            logWriter.add_scalar("loss_train/l2d", loss_l2d, n_iter)
            logWriter.add_scalar("loss_train/nlm", loss_nlm, n_iter)
            logWriter.add_scalar("loss_train/tv", loss_tv3d, n_iter)
            # logWriter.add_scalar("loss_train/possion", possion, n_iter)
            # logWriter.add_scalar("loss_train/sparsity", loss_sparisty, n_iter)
            # train_loss[items[0]].append(loss.data.cpu().numpy())
            # train_loss[items[1]].append(loss_kl.data.cpu().numpy())
            # train_loss[items[2]].append(loss_l2.data.cpu().numpy())
            # train_loss[items[3]].append(loss_nlm.data.cpu().numpy())

        if n_iter % params.num_save == 0: 
        # if n_iter % 1 == 0: 
            print("Sart validation...")
            with torch.no_grad():
                val_loss, logWriter = validate(model, val_loader, n_iter, val_loss, params, logWriter,logging)
                # logWriter = test_on_align_fk_10min(model, n_iter, logWriter, params)
                logWriter = test_on_align_cvpr(model, n_iter, logWriter, params)
                logWriter = test_on_align_fk(model, n_iter, logWriter, params)
            # save model states
            print("Validation complete!")
            save_checkpoint(n_iter, epoch, model, optimer,
            file_path=params.model_dir+"/epoch_{}_{}.pth".format(epoch, n_iter))
            print("Checkpoint saved!")
    
    return model, optimer, n_iter, train_loss, val_loss, logWriter

