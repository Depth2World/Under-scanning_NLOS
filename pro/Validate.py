# The validation function
import sys
sys.path.append("../util/")
import numpy as np 
import torch
from tqdm import tqdm
import util.SetDistTrain as utils
from pro.Loss import criterion_KL, criterion_L2, criterion_TV,NLM,criterion_L2_var
import scipy.io as scio
import os 
from metric import RMSE, PSNR, SSIM, AverageMeter
from models import embedfeature
import cv2
import torch.nn.functional as F
from sklearn.metrics import accuracy_score as ACC


lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
dtype = torch.cuda.FloatTensor
items = ["ALL", "KL", "L2intensity", "nlm", "spa", "TV","L2depth"]
metric_list = ['rmse', 'psnr', 'ssim','acc']
val_metrics = {k: AverageMeter() for k in metric_list}

def tv_3d(x):
    return torch.sum(torch.abs(x[ :, :, :, :, :-1] - x[ :, :, :, :, 1:])) + \
           torch.sum(torch.abs(x[ :, :, :, :-1, :] - x[ :, :, :, 1:, :])) + \
           torch.sum(torch.abs(x[ :, :, :-1, :, :] - x[ :, :, 1:, :, :])) 
           
           
def validate(model, val_loader, n_iter, val_loss, params, logWriter, logging):
    
    model.eval()
    l_all = []
    l_l2 = []
    l_l2d = []
    l_kl = []
    l_3tv = []
    l_nlm = []
    rmse = RMSE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()
    
    for sample in tqdm(val_loader):
        M_mea, raw_mea, dep_gt, img_gt= sample["ds_meas"].type(dtype), sample["raw_meas"].type(dtype), sample["dep_gt"].type(dtype), sample["img_gt"].type(dtype)
        
        up_M_mea, re_vlo, inten_re, target, dep_re, targetd = model(M_mea, img_gt, dep_gt)
        int_re, target, dep_re, targetd = inten_re.squeeze(1),target.squeeze(1), dep_re.squeeze(1), targetd.squeeze(1)
        loss_kl = criterion_L2(up_M_mea, raw_mea).data.cpu().numpy()
        loss_l2 = criterion_L2(int_re, target).data.cpu().numpy()
        loss_l2d = criterion_L2(dep_re, targetd).data.cpu().numpy()
        
        _,c,_,_,_ = re_vlo.shape
        loss_nlm = 0
        if c >= 1 :
            for ch in range(c):
                vlo = re_vlo[:,ch,...]
                loss_nlm += NLM(vlo)
        else:
            loss_nlm = NLM(re_vlo)
            
        loss_nlm = loss_nlm.data.cpu().numpy()
        loss_tv3d = tv_3d(re_vlo).data.cpu().numpy()
        
        loss =  params.kl_scale * loss_kl +  loss_l2 + 0 * loss_l2d + params.nlm_scale * loss_nlm + params.tv3d_scale * loss_tv3d #+ possion #+ 1e-5 * loss_nlm
        l_all.append(loss)
        l_l2.append(loss_l2)
        l_l2d.append(loss_l2d)
        l_kl.append(loss_kl)
        l_nlm.append(loss_nlm)
        l_3tv.append(loss_tv3d)
        
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
        
        metric_dict = {
            'rmse': rmse(int_re.clamp(0,1), target).cpu(),
            'psnr': psnr(int_re.clamp(0,1), target).cpu(),
            'ssim': ssim(int_re.clamp(0,1), target).cpu(),
            'acc': acc_sum,
        }
        for k in metric_list:
            val_metrics[k].update(metric_dict[k].item())
    # log the val losses
    if utils.is_main_process():
        logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
        logWriter.add_scalar("loss_val/l2", np.mean(l_l2), n_iter)
        logWriter.add_scalar("loss_val/l2", np.mean(l_l2d), n_iter)
        logWriter.add_scalar("loss_val/kl", np.mean(l_kl), n_iter)
        logWriter.add_scalar("loss_val/nlm", np.mean(loss_nlm), n_iter)
        logWriter.add_scalar("loss_val/3dtv", np.mean(loss_tv3d), n_iter)
        for k in metric_list:
            logWriter.add_scalars(k, {'test': val_metrics[k].item()}, n_iter)
            
        logWriter.add_images("inten_rec", int_re.clamp(0,1), n_iter,dataformats="NCHW")
        logWriter.add_images("inten_gt", target, n_iter,dataformats="NCHW")
        
        # val_loss[items[0]].append(np.mean(l_all))
        # val_loss[items[1]].append(np.mean(l_kl))
        # val_loss[items[2]].append(np.mean(l_l2))
        # val_loss[items[3]].append(np.mean(l_nlm))
        
        
        log_str = 'N_iter_Test_{} | '.format(n_iter)
        for k in val_metrics:
            log_str += '{:s} {:.5f} | '.format(k, val_metrics[k].item())
        if utils.is_main_process():
            logging.info(log_str)
            
    return val_loss, logWriter



def test_on_align_fk(model, n_iter, logWriter,args):
    model_dict = model.state_dict()
    test_model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.0096 * 2,views=1,wall_size=2, sp_ds_scale=args.sp_ds_scale)
    
    test_model = torch.nn.DataParallel(test_model)
    test_model.load_state_dict(model_dict)
    
    out_path = args.model_dir + '/test_on_fk/'
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True) 
    
    rw_path = '/data2/yueli/dataset/NLOS_RW/align_fk_256_512/'
    files = os.listdir(rw_path)
    all_file = []
    # output_path = []
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        # output_path.append(os.path.join(out_path, fi))
        
    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])['final_meas']
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 256, 256, 512  32ps
        M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        
         # spatial sample 
        ds = args.sp_ds_scale # 128 // 4
        if args.mask:
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
            
        with torch.no_grad():
            test_model.eval()
            vlo, im_re = model(M_mea)
            front_view = im_re.detach().cpu().numpy()[0, 0]
            cv2.imwrite(out_path + f'n_iter{n_iter}_{i}.png', (front_view / np.max(front_view))*255)
            
    if utils.is_main_process():
        print('validate on fk ')
    return logWriter
    


def test_on_align_fk_10min(model, n_iter, logWriter,args):
    model_dict = model.state_dict()
    test_model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.0096 * 2,views=1,wall_size=2, sp_ds_scale=args.sp_ds_scale)
    
    test_model = torch.nn.DataParallel(test_model)
    test_model.load_state_dict(model_dict)
    
    out_path = args.model_dir + '/test_on_fk10min/'
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True) 
    
    rw_path = '/data2/yueli/dataset/NLOS_RW/align_fk_256_512_meas_10min'
    files = os.listdir(rw_path)
    all_file = []
    # output_path = []
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        # output_path.append(os.path.join(out_path, fi))
        
    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])['final_meas']
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 256, 256, 512  32ps
        M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  

         # spatial sample 
        ds = args.sp_ds_scale # 128 // 4
        if args.mask:
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
            
        with torch.no_grad():
            test_model.eval()
            vlo, im_re = model(M_mea)
            front_view = im_re.detach().cpu().numpy()[0, 0]
            cv2.imwrite(out_path + f'n_iter{n_iter}_{i}.png', (front_view / np.max(front_view))*255)
            
    if utils.is_main_process():
        print('validate on fk 10min')
    return logWriter
    


def test_on_align_cvpr(model, n_iter, logWriter, args):
    model_dict = model.state_dict()
    test_model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.0096 * 2,views=1,wall_size=2, sp_ds_scale=args.sp_ds_scale)
    
    test_model = torch.nn.DataParallel(test_model)
    test_model.load_state_dict(model_dict)
    
    out_path = args.model_dir + '/test_on_xu/'
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True) 
    rw_path = '/data2/yueli/dataset/NLOS_RW/cvpr2023_data/'
    files = os.listdir(rw_path)
    all_file = []
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    for i in range(len(all_file)):
        transient_data = scio.loadmat(all_file[i])
        transient_data = transient_data['data'].transpose([1,0,2])  #sig final_meas measlr
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,128,128,-1])   # 1, 1, 64, 64,2048  8ps
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        # spatial sample 
        ds = args.sp_ds_scale # 128 // 4
        if args.mask:
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

        with torch.no_grad():
            test_model.eval()
            vlo, im_re = model(M_mea)
            front_view = im_re.detach().cpu().numpy()[0, 0]
            cv2.imwrite(out_path + f'n_iter{n_iter}_{i}.png', (front_view / np.max(front_view))*255)
    
    if utils.is_main_process():
        print('validate on xu_old')
        
    return logWriter
    