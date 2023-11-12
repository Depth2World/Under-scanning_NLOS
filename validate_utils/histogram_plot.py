import scipy.io as scio
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt


spatial = 32

interplote_root = f'/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_data/interp128/sptial{spatial}'
interplote_all = []
files = os.listdir(interplote_root)
for fi in files:
        fi_d = os.path.join(interplote_root, fi)
        interplote_all.append(fi_d)
        print(fi_d)
print('interploate file')

pred_root = f'/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_data/pred_mea128/sptial{spatial}_66000'
pred_all = []
for fi in files:
        fii = fi[:-4] + '_mea.mat'
        fi_d = os.path.join(pred_root,fii)
        pred_all.append(fi_d)
        print(fi_d)
print('pred file')
        


for i in range(len(files)):
    histogram_path = '/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_data/histogram_com_200/' + f'{spatial}to128/' + files[i][:-4]
    if not os.path.exists(histogram_path):
        os.makedirs(histogram_path, exist_ok=True)  
         
    ### load inteploate transient data    
    transient_data = scio.loadmat(interplote_all[i])
    interploate_transient = transient_data['resized_mea']
    # print(interploate_transient.shape)
    
    ### load pred transient data    
    transient_data = scio.loadmat(pred_all[i])
    pred_transient = transient_data['resized_mea']
    # print(interploate_transient.shape)
    
    ### load gt transient data    
    gt_file = files[i][:-4].replace(f'{spatial}to128','128to128')
    print(gt_file)
    gt_path = '/data/yueli/nlos_sp_output/nips_cs/traditional_algos/fk_data/interp128/sptial128/' + gt_file
    transient_data = scio.loadmat(gt_path)
    gt_transient = transient_data['resized_mea']
    print(gt_transient.shape)

    for t in range(0,128,1):
        i_his = interploate_transient[t,t,:]
        p_his = pred_transient[t,t,:]
        g_his = gt_transient[t,t,:]
        # print(i_his.shape)
        
        # colors=['orange', 'green', 'red'] # 'purple',
        # plt.gca().set_prop_cycle(color=colors)

        plt.plot(i_his,label='interp',color = 'b')
        plt.plot(p_his,label='pred',color = 'g')
        plt.plot(g_his,label='raw',color = 'r')
        
        plt.axis('tight') 
        plt.xlim(200,300)
        # plt.ylim(np.min(y.cumsum())- 1, np.max(y.cumsum()) + 1)
        # plt.xlabel("X",fontsize=13)
        # plt.ylabel("Y",fontsize=13)
        plt.legend(fontsize=16,loc='upper left')
        # plt.title("Plot Mutiple lines in Matplotlib",fontsize=15)
        fig_path = histogram_path + f'/{t}_{t}_' + files[i][:-4] + '.png'
        plt.savefig(fig_path) #, dpi = 2000
        plt.clf()
# transient_data = scio.loadmat(all_file[i])
#         # transient_data = transient_data['data'].transpose([2,1,0])  #sig final_meas measlr
#         transient_data = transient_data['resized_mea']