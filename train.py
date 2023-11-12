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
import logging
from util.LFEDataset import LFEDataset, NLOSDataset
from util.ParseArgs import get_args_parser
from util.SaveChkp import save_checkpoint
import util.SetDistTrain as utils
from pro.Train import train
from models import embedfeature
cudnn.benchmark = True


def update_parse_args(opt):
    today = datetime.today()
    opt.model_dir += opt.model_name +"_"+ str(today.year)+"_"+str(today.month)+str(today.day)
    # mkdirs if necessary
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir, exist_ok=True)
    # save args to files
    if opt.param_store:
        args_dict = opt.__dict__
        config_bk_pth = opt.model_dir + "/config_bk.txt"
        with open(config_bk_pth, "w") as cbk_pth:
            cbk_pth.writelines("------------------Start------------------"+ "\n")
            for key, valus in args_dict.items():
                cbk_pth.writelines(key + ": " + str(valus) + "\n")
            cbk_pth.writelines("------------------End------------------"+ "\n")
        print("Config file load complete! \nNew file saved to {}".format(config_bk_pth))
    return opt



def main(opt):
    
    # parse arguments
    opt = update_parse_args(opt)
    # set distribution training (add args.distributed = True)
    utils.init_distributed_mode(opt)
    device = torch.device(opt.device)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print(opt)
    print("Current main process GPUs: {}".format((opt.loc_rank)))
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Number of Encoder-Decoders: {}".format(opt.num_coders))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    

    # load data
    print("Loading training and validation data...")
    # build dataset
    # fix the seed in dataset building for reproducibility
    seed = opt.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(opt.model_dir + '/train.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    logging.info('='*80)
    logging.info(f'Start of experiment: {opt.model_name}')
    logging.info('='*80)
    logging.info("Loading training and validation data...")
    
    if opt.dataset=='bike':
        folder_path = ['/data2/yueli/dataset/LFE_dataset/bike']
        shineness = [0]
        logging.info(folder_path[0])
        train_data = LFEDataset(root=folder_path, # dataset root directory
                                shineness=shineness,
                                for_train=True,
                                ds=1,               # temporal down-sampling factor
                                clip=512,           # time range of histograms
                                size=256,           # measurement size (unit: px)
                                scale=1,            # scaling factor (float or float tuple)
                                background=[0.05,2],# background noise rate (float or float tuple)
                                target_size=128,    # target image size (unit: px)
                                target_noise=0.01,  # standard deviation of target image noise
                                color='gray',       # color channel(s) of target image
                                sp_ds=opt.sp_ds_scale, # spatial resolution downsample
                                mask=opt.mask)         # mea * mask or not 
        
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
                                sp_ds=opt.sp_ds_scale, # spatial resolution downsample
                                mask=opt.mask)         # mea * mask or not 
    else:
        root_path = '/data2/yueli/dataset/LFE_dataset/NLOS_bike_allviews_processed'
        logging.info(root_path)
        train_data = NLOSDataset(root=root_path,  # dataset root directory
                                split=True, # data split ('train', 'val')
                                ds=1, # temporal down-sampling factor
                                clip=512, # time range of histograms
                                size=128, # measurement size (unit: px)
                                d_s=opt.sp_ds_scale, # scaling factor (float or float tuple)
                                background=0, #0.01529, # background noise rate (float or float tuple)
                                target_size=128, # target image size (unit: px)
                                target_noise=0.01, # standard deviation of target image noise
                                color='gray',
                                mask=opt.mask) # color channel(s) of target image
            
        val_data = NLOSDataset(root=root_path,  # dataset root directory
                                split=False, # data split ('train', 'val')
                                ds=1, # temporal down-sampling factor
                                clip=512, # time range of histograms
                                size=128, # measurement size (unit: px)
                                d_s=opt.sp_ds_scale, # scaling factor (float or float tuple)
                                background=0, #0.01529, # background noise rate (float or float tuple)
                                target_size=128, # target image size (unit: px)
                                target_noise=0.01, # standard deviation of target image noise
                                color='gray',
                                mask=opt.mask) # color channel(s) of target image
    
    logging.info(len(train_data))
    logging.info(len(val_data))
    if opt.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = DistributedSampler(train_data,num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = SequentialSampler(val_data)
    else:
        train_sampler = RandomSampler(train_data)
        val_sampler = SequentialSampler(val_data)

    train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, sampler=val_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True)
    print("Load training and validation data complete!")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    # build network and move it multi-GPU
    print("Constructing Models...")
    
    model = embedfeature.EmbedFeatureModel_MUL_gray_former_mask_refine(basedim = 3, in_ch=1,out_ch=1,spatial=64,tlen=256,bin_len=0.02,views=1,wall_size=2, sp_ds_scale=opt.sp_ds_scale)
    
    model.to(device)
    # logging.info(model)
    if opt.distributed:
        model = DDP(model, device_ids=[opt.loc_rank],find_unused_parameters=True)
        logging.info("Models constructed complete! Paralleled on {} GPUs".format(torch.cuda.device_count()))
    else:
        logging.info("Models constructed complete on SINGLE GPU!")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Numbers of parameters are: {}".format(num_params))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    
    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.opter == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=opt.lr_rate, weight_decay=opt.weit_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=opt.lr_rate)

    n_iter = 0
    start_epoch = 1
    items = ["ALL", "KL", "L2intensity", "nlm", "spa", "TV","L2depth"]
    train_loss = {items[0]: [], items[1]: [], items[2]: [], items[3]: [], items[4]: [], items[5]: []}
    val_loss = {items[0]: [], items[1]: [], items[2]: [], items[3]: [], items[4]: [], items[5]: []}
    logWriter = SummaryWriter(opt.model_dir + "/")
    logging.info("Parameters initialized")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    if opt.resume:
        if os.path.exists(opt.resmod_dir):
            print("Loading checkpoint from {}".format(opt.resmod_dir))
            checkpoint = torch.load(opt.resmod_dir, map_location="cpu")
            
            # load start epoch
            try:
                start_epoch = checkpoint['epoch']
                print("Loaded and update start epoch: {}".format(start_epoch))
            except KeyError as ke:
                start_epoch = 1
                print("No epcoh info found in the checkpoint, start epoch from 1")
            
            # load iter number
            try:
                n_iter = checkpoint["n_iter"]
                print("Loaded and update start iter: {}".format(n_iter))
            except KeyError as ke:
                n_iter = 0
                print("No iter number found in the checkpoint, start iter from 0")

            # load learning rate
            try:
                opt.lr_rate = checkpoint["lr"]
            except KeyError as ke:
                print("No learning rate info found in the checkpoint, use initial learning rate:")
            
            # load model params
            model_dict = model.state_dict()
            try:
                ckpt_dict = checkpoint['state_dict']
                # for k in ckpt_dict.keys():
                #     model_dict.update({k[7:]: ckpt_dict[k]})
                # model.load_state_dict(model_dict)
                model.load_state_dict(ckpt_dict)
                print("Loaded and update model states!")
            except KeyError as ke:
                print("No model states found!")
                sys.exit("NO MODEL STATES")

            # simple model dict load methods (2021.12.6)
            # model_wo_ddp = model.module
            # model_wo_ddp.load_state_dict(checkpoint['state_dict'])

            # load optimizer state
            for g in optimizer.param_groups:
                g["lr"] = opt.lr_rate
                print("Loaded learning rate!")
            
            print("Checkpoint load complete!!!")

        else:
            print("No checkPoint found at {}!!!".format(opt.resmod_dir))
            sys.exit("NO FOUND CHECKPOINT ERROR!")

    else:
        print("Do not resume! Use initial params and train from scratch.")

    # start training 
    print("Start training...")
    for epoch in range(start_epoch, opt.num_epoch):
        print("Epoch: {}, LR: {}".format(epoch, optimizer.param_groups[0]["lr"]))

        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model, optimizer, n_iter, train_loss, val_loss, logWriter = \
            train(model, train_loader, val_loader, optimizer, \
                epoch, n_iter, train_loss, val_loss, opt, logWriter, logging)
            
        save_checkpoint(n_iter, epoch, model, optimizer,\
            file_path=opt.model_dir+"/epoch_{}_{}_END.pth".format(epoch, n_iter))

        print("End of epoch: {}. Checkpoint saved!".format(epoch))


if __name__=="__main__":
    opt = get_args_parser()
    main(opt)



