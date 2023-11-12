# This is the file for loss functions
import numpy as np 
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    # grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    # the computed grad's edge is useless, remove them
    grad_x = grad_x[:,:,2:-2,2:-2]
    grad_y = grad_y[:,:,2:-2,2:-2]
    out = torch.cat((torch.reshape(grad_y,(N,C,-1)), torch.reshape(grad_x,(N,C,-1))), dim=1)
    return out


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    def forward(self, inpt, tar):
        grad_tar = imgrad_yx(tar)
        grad_inpt = imgrad_yx(inpt)
        loss = torch.sum(torch.mean(torch.abs(grad_tar - grad_inpt)))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, inpt, target):
        loss = nn.L1Loss()(torch.log(inpt), torch.log(target))
        return loss

        
###########################################
# the set of loss functions
criterion_GAN = nn.MSELoss()
criterion_KL = nn.KLDivLoss()

# inpt, target: [batch_size, 1, h, w]
criterion_L1 = nn.L1Loss()
criterion_L1log = L1_log()
criterion_grad = GradLoss()



def NLM(es,block_size=3):
    p = block_size // 2
    es_pad = torch.nn.functional.pad(es, (p,p,p,p), mode='replicate')
    es_uf = torch.nn.functional.unfold(es_pad, kernel_size=block_size)
    es_uf = es_uf.view(es.shape[0], es.shape[1], -1, es.shape[2], es.shape[3])

    weight = create_window(window_size=block_size, channel=1)
    weight = weight.view(1,-1).to(es.device)
    es_uf = es_uf.permute(0,2,1,3,4)
    weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(es_uf.shape)

    neibor = torch.mul(es_uf, weight)
    neibor_sum = torch.sum(neibor, dim=1, keepdim=False)
    return criterion_L2(es, neibor_sum)



def criterion_TV(inpt):
    return torch.sum(torch.abs(inpt[:, :, :, :-1] - inpt[:, :, :, 1:])) + \
           torch.sum(torch.abs(inpt[:, :, :-1, :] - inpt[:, :, 1:, :]))


def criterion_L2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))

def criterion_L2_var(est, gt, var):
    criterion = nn.MSELoss()
    return torch.mean(torch.exp(-var) * criterion(est, gt)) + 0.1 * torch.mean(var)

def criterion_KL_noise(est, gt):
    h = np.random.randint(32, size=2)
    w = np.random.randint(32, size=2)

    loss = criterion_KL(est[:,:,:, h[0], w[0]], gt[:, :, :, h[1], w[1]])

    return loss


def criterion_REG(inpt):
    # the inpt should be a 3D volume， B, D, H, W
    # we may need to change the params to with grad
    B, D, H, W = inpt.squeeze(1).size()
    inpt_3D = torch.reshape(inpt.squeeze(1), (B, D, H*W)).permute(0,2,1)
    d_bh = []

    for blk in range(B):
        d_dp = []
        for pix in range(H*W):
            temp = torch.squeeze(inpt_3D[blk, pix, :], 0)
            nonzs = torch.nonzero(temp).squeeze(1)
            d_dp.append(nonzs[-1] - nonzs[0])
        
        d_bh.append(torch.sum(torch.from_numpy(np.asarray(d_dp)).float()))

    return torch.mean(torch.from_numpy(np.asarray(d_bh)).float().cuda()) / 1024


def criterion_REG2(inpt):
    # the inpt should be a 3D volume， B, D, H, W
    # we may need to change the params to with grad
    B, D, H, W = inpt.squeeze(1).size()
    inpt_3D = torch.reshape(inpt.squeeze(1), (B, D, H*W)).permute(0,2,1)
    d_bh = []

    for blk in range(B):
        d_dp = []
        for pix in range(H*W):
            temp = torch.squeeze(torch.squeeze(inpt_3D[blk, pix, :], 0), 0)
            thrd = torch.min(temp) + (torch.max(temp) - torch.min(temp)) * 0.9
            temp[temp<=thrd] = 0
            nonzs = torch.nonzero(temp).squeeze(1)
            d_dp.append(nonzs[-1] - nonzs[0])
        
        d_bh.append(torch.sum(torch.FloatTensor(d_dp)))

    res = torch.mean(torch.FloatTensor(d_bh).cuda()) / 1024
    res.requires_grad=True

    return res


def criterion_NONZINDEX_MEAN(inpt):
    # this is to return the nonzero index in the tensor
    # the inpt should be a 3D volume， B, D, H, W
    B, D, H, W = inpt.size()
    inpt_3D = torch.reshape(inpt, (B, D, H*W)).permute(0,2,1)
    out_3D = torch.zeros([B, H*W, 1000], dtype=torch.float, requires_grad=True).cuda()

    for blk in range(B):
        for pix in range(H*W):
            _, out_3D[blk, pix, :] = torch.topk(torch.squeeze(torch.squeeze(inpt_3D[blk, pix, :], 0), 0), 1000)

    res = torch.reshape(out_3D, (B, H, W, 1000)).permute(0, 3, 1, 2)
    return torch.mean(res, dim=1, keepdim=True)


def criterion_NONZINDEX_STD(inpt):
    # this is to return the nonzero index in the tensor
    # the inpt should be a 3D volume， B, D, H, W
    B, D, H, W = inpt.size()
    inpt_3D = torch.reshape(inpt, (B, D, H*W)).permute(0,2,1)
    out_3D = torch.zeros([B, H*W, 1000], dtype=torch.float, requires_grad=True).cuda()

    for blk in range(B):
        for pix in range(H*W):
            _, out_3D[blk, pix, :] = torch.topk(torch.squeeze(torch.squeeze(inpt_3D[blk, pix, :], 0), 0), 1000)

    res = torch.reshape(out_3D, (B, H, W, 1000)).permute(0, 3, 1, 2)
    return torch.std(res, dim=1, keepdim=True)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    """
    Useage: 1-SSIMLoss(x,y)
    """
    def __init__(self, window_size=7, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def criterion_SSIM(est, gt):
    criterion = SSIMLoss()

    return criterion(est, gt)


def ssim(img1, img2, window_size=7, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load("./VGG_PreModel/vgg19-dcbb9e9d.pth"))
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return h_relu4
        # return h_relu1


class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()   #original is .cuda(gpu_id), may cause occupancy imbalance in multi-GPU training
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        # for i in range(len(x_vgg)):
        loss =  self.criterion(x_vgg, y_vgg.detach())
        return loss


def criterion_VGG(est, gt):
    criterion = VGGLoss()

    return criterion(est.repeat(1, 3, 1, 1), gt.repeat(1, 3, 1, 1))
