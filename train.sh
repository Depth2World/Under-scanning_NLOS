#!/bin/bash
# hyperparameters
# --sp_ds_scale  -> downsample ratio (raw resolution is 128 *128)
# --model_dir  ->  output path

python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
--bacth_size 1 \
--model_name  Under_scanning_ds8 \
--dataset bike \
--num_save 300 \
--sp_ds_scale 8 \
--model_dir /data2/yueli/nlos_cs_output/master \
--po_scale 0 \
--kl_scale 1 \
--nlm_scale 1e-5 \
--tv3d_scale 1e-6 \
--num_epoch 50 \
--mask
