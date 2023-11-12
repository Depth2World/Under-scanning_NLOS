from enum import EnumMeta
import os 
import glob
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F


def list_file_path(root_path,fortrain=True,shineness=[0]):
    modeldirs = []
    testmodeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    root_path = [root_path]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            # print(modeldir_all)
            
            testdir_all = []
            with open('/data2/yueli/code/nlos_cs_nips2023/util/A100_allbike_testlist.txt', 'r', encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = ann.strip('\n')       #去除文本中的换行符
                    testdir_all.append(ann)
            # print(len(testdir_all))
            traindir_all = [ dir for dir in modeldir_all if dir not in testdir_all]
            # print(len(traindir_all))

            for modeldir in traindir_all:# modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:11]   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDPS*.mp4' % (dir))
                    # path = glob.glob('%s/video-confocalspadPDP*.hdr' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    de.append(path)
            for modeldir in testdir_all: #modeldir_all[250:]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))[:1] #[:11]   # file_path/0/XXX/shinexxx_rot xxx
                testmodeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDPS*.mp4' % (dir))
                    # path = glob.glob('%s/video-confocalspadPDP*.hdr' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    test_de.append(path)
    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': testmodeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample
    

def list_file_path_bike(root_path,fortrain=TRUE,shineness=[0]):
    modeldirs = []
    test_modeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    de.append(path)
                    
            for modeldir in modeldir_all[250:]:
                # print(modeldir)
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:1]   # file_path/0/XXX/shinexxx_rot xxx
                test_modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    test_de.append(path)
                    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': test_modeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample
    

    
def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)



class LFEDataset(Dataset):

    def __init__(
        self, 
        root,               # dataset root directory
        shineness,              # data split ('train', 'val')
        for_train=True,
        ds=1,               # temporal down-sampling factor
        clip=512,           # time range of histograms
        size=128,           # measurement size (unit: px)
        scale=1,            # scaling factor (float or float tuple)
        background=0,       # background noise rate (float or float tuple)
        target_size=128,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        color='gray',       # color channel(s) of target image
        sp_ds=8,            # spatial resolution downsample,
        mask = False
    ):
        super(LFEDataset, self).__init__()
        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.target_size = target_size
        self.target_noise = target_noise
        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.data_list = list_file_path_bike(root,for_train,shineness)
        self.color = color
        self.sp_ds = sp_ds
        self.mask = mask
        
    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr','mp4')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.reshape(-1, x.shape[1], x.shape[1], 1)  
                x = x.transpose(3, 0, 1, 2)  # 1 600 256 256 
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
                        ims.append(imgray)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                data = np.array(ims, dtype=np.float32) / 255.0   
                raw = data[None] #.reshape(1,data.shape[0],data.shape[1],data.shape[2])    # 1 600 256  256 
                # raw = (raw[:,:,::2,:] + raw[:,:,1::2,:])/2    
                # raw = (raw[:,:,:,::2] + raw[:,:,:,1::2])/2
                raw = (raw[:,:,::2,:] + raw[:,:,1::2,:])    
                raw = (raw[:,:,:,::2] + raw[:,:,:,1::2])
                
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
        
        # clip temporal range
        raw = raw[:, :self.clip]                                # (1/3, t, h, w)
        
        # temporal down-sampling
        if self.ds > 1:
            c, t, h, w = raw.shape
            assert t % self.ds == 0
            raw = raw.reshape(c, t // self.ds, self.ds, h, w)
            raw = raw.sum(axis=2)
            
        # spatial sub-sampling
        if self.mask:
            raw_ = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
            # raw = self.add_noise(raw_)
            raw = raw_
            # print(raw.shape)
            mask = torch.zeros_like(raw) # 1 t h w
            for index_i in range(self.sp_ds//2, 128, self.sp_ds):
                for index_j in range(self.sp_ds//2, 128, self.sp_ds):
                    mask[:,:,index_i,index_j] = 1
            x =  raw * mask
        else:
            if self.sp_ds >= 1:
                    c, t, h, w = raw.shape
                    raw_ = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
                    # raw = self.add_noise(raw_)
                    raw = raw_
            x = raw[:,:,self.sp_ds//2::self.sp_ds,self.sp_ds//2::self.sp_ds]
            x = F.interpolate(x,size=(self.target_size,) * 2) 
        return x, raw_     
                
                
    def add_noise(self,transient):
        # transient belong to 0~1  
        noise_b = torch.rand_like(transient) * 0.1
        transient += noise_b
        return transient  

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x.reshape( x.shape[1], x.shape[1], 1)   #  256 256 1 1 
                    x = cv2.resize(x,(self.target_size,self.target_size))
                    x = x.reshape( x.shape[1], x.shape[1], 1, 1)   #  128 128 1 1 
                    x = x / np.max(x)
                else:
                    x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 1/3)       #  256 256 1 1
        x = x.permute(2, 3, 0, 1)                               # (v, 1/3, h, w)       #  1  1  256 256 
        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data'] 
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)        #   meters  fe 0.597m
                x = x[..., 0]
                x = 1 - x.clip(0,1)  
                x = x.clip(0,1)
            x = cv2.resize(x, (self.target_size,self.target_size))          # (h, w)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x.transpose(2,3,0,1)
        x = torch.from_numpy(x.astype(np.float32))              # (v, 1, h, w)
        return x
    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas, raw_meas = self._load_meas(idx) #self.transform()  #   the hyperparameter [0.05 2] has err
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'ds_meas': meas, 'raw_meas': raw_meas, 'dep_gt': depths, 'img_gt': images}
        return sample



class NLOSDataset(Dataset):
    def __init__(
        self, 
        root,               # dataset root directory
        split=True,              # data split ('train', 'val')
        ds=1,               # temporal down-sampling factor
        clip=512,           # time range of histograms
        size=256,           # measurement size (unit: px)
        d_s=1,            # scaling factor (float or float tuple)
        background=0.1529,       # background noise rate (float or float tuple)
        target_size=256,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        color='gray',        # color channel(s) of target image
        mask = False,
    ):
        super(NLOSDataset, self).__init__()

        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.sp_ds = d_s
        self.transform = get_transform(background)
        self.target_size = target_size
        self.target_noise = target_noise

        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.mask = mask
        self.color = color
        self.split = split
        self.data_list = list_file_path(self.root,self.split)
        print(self.split,len(self.data_list['Mea']))


    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        
        assert ext in ('mat', 'hdr', 'mp4')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                x = x.reshape(-1, x.shape[1], x.shape[1], 1)
                raw = x.transpose(3, 0, 1, 2)
                raw = (raw[:,:,::2,:] + raw[:,:,1::2,:])#/2
                raw = (raw[:,:,:,::2] + raw[:,:,:,1::2])#/2
                # print(x.shape)
                
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     ## train gray need use
                        ims.append(frame)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0  
                raw = raw.transpose(3,0,1,2)   # 3 600 256 256  
                raw = (raw[:,:,::2,:] + raw[:,:,1::2,:])/2
                raw = (raw[:,:,:,::2] + raw[:,:,:,1::2])/2
                
            if raw.ndim == 3:
                raw = raw[None]
                
            # temporal down-sampling
            if self.ds > 1:
                c, t, h, w = raw.shape
                assert t % self.ds == 0
                raw = raw.reshape(c, t // self.ds, self.ds, h, w)
                raw = raw.sum(axis=2)

            # clip temporal range
            raw = raw[:, :self.clip] 
            if raw.shape[0] == 3:
                if self.color == 'gray':
                    raw = 0.299 * raw[0:1] + 0.587 * raw[1:2] + 0.114 * raw[2:3]
                elif self.color == 'r': raw = raw[0:1]
                elif self.color == 'g': raw = raw[1:2]
                elif self.color == 'b': raw = raw[2:3]
            else:
                if self.color == 'rgb':
                    raw = np.tile(raw, [3, 1, 1, 1])
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))

            
        # spatial sub-sampling
        if self.mask:
            raw_ = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
            raw = self.add_noise(raw_)
            # print(raw.shape)
            mask = torch.zeros_like(raw) # 1 t h w
            for index_i in range(self.sp_ds//2, 128, self.sp_ds):
                for index_j in range(self.sp_ds//2, 128, self.sp_ds):
                    mask[:,:,index_i,index_j] = 1
            x =  raw * mask
        else:
            if self.sp_ds >= 1:
                    c, t, h, w = raw.shape
                    raw_ = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
                    raw = self.add_noise(raw_)
                    # raw = torch.poisson(raw)
                    
                    # print(torch.max(raw))
            
            ## add noise for transient data
                
            x = raw[:,:,self.sp_ds//2::self.sp_ds,self.sp_ds//2::self.sp_ds]
            x = F.interpolate(x,size=(self.target_size,) * 2) 
        return x, raw_     
                
                
    def add_noise(self,transient):
        # transient belong to 0~1  
        noise_b = torch.rand_like(transient) * 0.2
        # noise_b = ( noise_b * 2 - 1 ) * 0.1 + 0.2 # 0.1 ~ 0.3
        transient += noise_b
        return transient  
    
    
    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['img']     # 26 256 256 3
                x = x.transpose(1,2,0,3).reshape(x.shape[1],x.shape[2],-1)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w, v*3)
                x = x / np.max(x)
                x = x.reshape(x.shape[0], x.shape[1], 26, 3)    # (h, w, v, 3)
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                if ext == 'hdr':
                    x = x.reshape(-1, x.shape[1], x.shape[1], 3)
                    x = x.transpose(1, 2, 0, 3)
                    x = x.reshape(*x.shape[:2], -1)
                else:
                    x = x / 255
                    
            if self.color == 'gray':
                x = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
            elif self.color == 'r': x = x[..., 0:1]
            elif self.color == 'g': x = x[..., 1:2]
            elif self.color == 'b': x = x[..., 2:3]
        except:
            raise ValueError('image loading failed: {:s}'.format(path))
       
        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 3)
        x = x.permute(2, 3, 0, 1)                               # (v, 1/3, h, w)
        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['depth']   # 26 256 256 3
                x = x.transpose(1,2,0,3).reshape(x.shape[1],x.shape[2],-1)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w, v*3)
                x = x.clip(0,1)
                x = x.reshape(x.shape[0], x.shape[1], 26, 3)    # (h, w, v, 3)
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = x[..., 0]
                x = x.reshape(-1, x.shape[1], x.shape[1])
                x = x.transpose(1, 2, 0)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        
        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 3)
        x = x.permute(2, 3, 0, 1)                               # (v, 1/3, h, w)
        x = torch.mean(x,dim=1,keepdim=True)
        return x
    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas,raw_meas = self._load_meas(idx)
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'ds_meas': meas, 'raw_meas': raw_meas, 'dep_gt': depths, 'img_gt': images}
        return sample


if __name__ == '__main__': 
    root_path = ['/data/yueli/dataset/bike']
    shineness = [0]
    #list_file_path(root_path,shineness)
    train_set = LFEDataset(root_path,shineness,False,1,512,256,1,[0.05,2],128,0,'gray',8)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=16) # drop_last would also influence the performance
    print(len(train_set))
    
    # folder_path = '/data/yueli/dataset/zip/NLOS_bike_allviews_color_processed'
    # train_data = NLOSDataset(folder_path,False,1,512,256,8,[0.05,0.03],128,0.000,'gray')
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=16) # drop_last would also influence the performance
    # print(len(train_data))
    
    for index,data in enumerate(train_loader):
        print(index)
        # mea = data['ds_meas']
        print(data['ds_meas'].shape,data['raw_meas'].shape,data['dep_gt'].shape,data['img_gt'].shape)   
        
        # LFE          torch.Size([4, 1, 512, 16, 16]) torch.Size([4, 1, 512, 128, 128]) torch.Size([4, 1, 1, 128, 128]) torch.Size([4, 1, 1, 128, 128])
        # NLOSDataset  torch.Size([4, 1, 512, 16, 16]) torch.Size([4, 1, 512, 128, 128]) torch.Size([4, 26, 1, 128, 128]) torch.Size([4, 26, 1, 128, 128])