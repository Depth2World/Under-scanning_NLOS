# The parse argments
import os
from pickle import TRUE
import sys
import argparse
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime


def get_args_parser():
    # build the arg_parser
    parser = argparse.ArgumentParser()
    # set path params
    parser.add_argument("--param_store", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default="debug") #  TST_Large_DP_final_model_upgrade
    parser.add_argument("--dataset", type=str, default='bike')
    parser.add_argument("--sp_ds_scale", type=int, default=8)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    parser.add_argument("--nlm_scale", type=float, default=0.1)
    parser.add_argument("--po_scale", type=float, default=0)
    parser.add_argument("--tv3d_scale", type=float, default=1e-6)
    parser.add_argument("--mask", action='store_true', help='default False')
    parser.add_argument("--model_dir", type=str, default="/data/yueli/nlos_sp_output/nips_cs/train_on_bike/")
    parser.add_argument("--seed", type=int, default=3407, help="seed for data loading")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resmod_dir", type=str, default=None) # ./unet_small_nyu_epoch_15_68800.pth 
    # set model params
    parser.add_argument("--bacth_size", type=int, default=4)
    parser.add_argument("--down_scale", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--num_save", type=int, default=300) #800
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_coders", type=int, default=1)
    parser.add_argument("--lr_rate", type=float, default=1e-4)
    parser.add_argument("--weit_decay", type=int, default=1e-4)
    parser.add_argument("--loss_weit", type=float, default=0.)
    parser.add_argument("--grad_clip", type=int, default=0.1, help="gradient clipping max norm")
    parser.add_argument("--noise_idx", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--drop_attn", type=float, default=0.)
    parser.add_argument("--drop_proj", type=float, default=0.)
    parser.add_argument("--drop_path", type=float, default=0.)
    # set optimization params
    parser.add_argument("--opter", type=str, default="adamw")
    parser.add_argument("--epo_warm", type=int, default=5)
    parser.add_argument("--epo_cool", type=int, default=5)
    # set distributed training params
    parser.add_argument("--dp_gpus", type=str, default="0,1,2,3")# 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1, help="number of total machines")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="env://", help="url used to setup the distributed training")

    return parser.parse_args()



