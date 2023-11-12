import torch.fft as fft
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.signal as ssi


def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'exp':
        return lambda x: torch.exp(x)
    elif actv == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif actv == 'tanh':
        return lambda x: torch.tanh(x)
    elif actv == 'softplus':
        return lambda x: torch.log(1 + torch.exp(x - 1))
    elif actv == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid activation function: {:s}'.format(actv)
        )

def make_norm2d(name, plane, affine=True):
    if name == 'batch':
        return nn.BatchNorm2d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm2d(plane, affine=affine)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )

def make_norm3d(name, plane, affine=True, per_channel=True):
    if name == 'batch':
        return nn.BatchNorm3d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm3d(plane, affine=affine)
    elif name == 'max':
        return MaxNorm(per_channel=per_channel)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )


class MaxNorm(nn.Module):
    """ Per-channel normalization by max value """
    def __init__(self, per_channel=True, eps=1e-8):
        super(MaxNorm, self).__init__()

        self.per_channel = per_channel
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): raw RSD output.

        Returns:
            x (float tensor, (bs, c, d, h, w)): normalized RSD output.
        """
        assert x.dim() == 5, \
            'input should be a 5D tensor, got {:d}D'.format(x.dim())

        if self.per_channel:
            x = F.normalize(x, p=float('inf'), dim=(-3, -2, -1))
        else:
            x = F.normalize(x, p=float('inf'), dim=(-4, -3, -2, -1))
        return x







class RSDBase(nn.Module):
    """ Rayleigh-Sommerfield diffraction kernel """
    def __init__(
        self, 
        t=512,              # time dimension of input volume
        d=512,               # depth dimension of output volume
        h=128,               # height dimension of input/output volume
        w=128,               # width dimension of input/output volume
        in_plane=6,         # number of input planes
        wall_size=2,        # wall size (unit: m)
        bin_len=0.02,       # distance covered by a bin (unit: m)
        zmin=0,             # min reconstruction depth w.r.t. the wall (unit: m)
        zmax=2,             # max reconstruction depth w.r.t. the wall (unit: m)
        scale_coef=1,       # scale coefficient for virtual wavelength
        n_cycles=4,         # number of cycles for virtual wavelet
        ratio=0.1,          # relative magnitude under which a frequency is discarded
        actv="linear",      # activation function
        norm="max",         # normalization function
        per_channel=True,   # if True, perform per-channel normalization
        affine=False,       # if True, apply a learnable affine transform in norm
        efficient=False,    # if True, use memory-efficient implementation
        **kwargs,
    ):
        super(RSDBase, self).__init__()
        assert t % 2 == 0, "time dimension must be even"

        self.t = t
        self.d = d
        self.h = h
        self.w = w
        self.in_plane = in_plane
        self.out_plane = in_plane

        self.wall_size = wall_size
        self.bin_len = bin_len
        self.zmin = zmin
        self.zmax = zmax

        self.scale_coef = scale_coef
        self.n_cycles = n_cycles
        self.ratio = ratio

        bin_resolution = bin_len / 3e8       # temporal bin resolution (unit: sec)
        sampling_freq = 1 / bin_resolution   # temporal sampling frequency

        # define virtual wave
        wall_spacing = wall_size / h         # sample spacing on the wall (unit: m)
        lambda_limit = 2 * wall_spacing      # smallest achievable wavelength
        wavelength = scale_coef * lambda_limit

        wave = self._define_wave(wavelength)
        fwave = np.abs(np.fft.fft(wave) / t)[:len(wave) // 2 + 1]
        coef_ratio = fwave / np.max(fwave)

        # retain spectrum [lambda - delta, lambda + delta]
        freq_idx = np.where(coef_ratio > ratio)[0]
        print(
            "{:d}/{:d} frequencies kept in RSD".format(
                len(freq_idx), len(fwave)
            )
        )
        freqs = sampling_freq * freq_idx / t
        omegas = 2 * np.pi * freqs           # angular frequencies
        coefs = fwave[freq_idx]              # weight cofficients

        # define RSD kernel
        zdim = np.linspace(zmin, zmax, d + 1)
        zdim = (zdim[:-1] + zdim[1:]) / 2    # mid-point rule

        rsd, tgrid = self._define_rsd(zdim, omegas)
        if not efficient:
            rsd = np.pad(rsd, ((0, 0), (0, 0), (0, h), (0, w))) # (o, d, h*2, w*2)
        frsd = np.fft.fft2(rsd)                                 # (o, d, h(*2), w(*2))
        
        # define phase term in IFFT
        omegas = omegas.reshape(-1, 1, 1, 1)                    # (o, 1, 1, 1)
        tgrid = (zdim / 3e8).reshape(1, -1, 1, 1)               # (1, d, 1, 1)
        phase = np.exp(1j * omegas * tgrid)                     # (o, d, h/1, w/1)

        # parameters associated with virtual wave
        freq_idx = torch.from_numpy(freq_idx)                   # (o,)
        self.register_buffer("freq_idx", freq_idx, persistent=False)

        coefs = torch.from_numpy(coefs.astype(np.float32))
        coefs = coefs.reshape(-1, 1, 1)                         # (o, 1, 1)
        self.register_buffer("coefs", coefs, persistent=False)

        # parameters associated with RSD propagation
        frsd = torch.from_numpy(frsd.astype(np.complex64))      # (o, d, h(*2), w(*2))
        self.register_buffer("frsd", frsd, persistent=False)
        
        phase = torch.from_numpy(phase.astype(np.complex64))    # (o, d, h, w)
        self.register_buffer("phase", phase, persistent=False)

        self.actv = make_actv(actv)
        self.norm = make_norm3d(norm, in_plane, affine, per_channel)

    def _define_wave(self, wavelength):
        # discrete samples of the virtual wavelet
        samples = round((self.n_cycles * wavelength) / self.bin_len)
        n_cycles = samples * self.bin_len / wavelength
        idx = np.arange(samples) + 1

        # complex-valued sinusoidal wave modulated by gaussian envelope
        sinusoid = np.exp(1j * 2 * np.pi * n_cycles * idx / samples)
        win = ssi.gaussian(samples, (samples - 1) / 2 * 0.3)
        wave = sinusoid * win

        # pad wave to the same length as time-domain histograms
        if len(wave) < self.t:
            wave = np.pad(wave, (0, self.t - len(wave)))
        return wave

    def _define_rsd(self, zdim, omegas):
        width = self.wall_size / 2
        ydim = np.linspace(width, -width, self.h + 1)
        xdim = np.linspace(-width, width, self.w + 1)
        ydim = (ydim[:-1] + ydim[1:]) / 2   # mid-point rule
        xdim = (xdim[:-1] + xdim[1:]) / 2
        [zgrid, ygrid, xgrid] = np.meshgrid(zdim, ydim, xdim, indexing="ij")

        # a grid of distance between wall center and scene points
        # (assume light source lies at wall center)
        dgrid = np.sqrt((xgrid ** 2 + ygrid ** 2) + zgrid ** 2) # (d, h, w)
        tgrid = zgrid / 3e8                                     # (d, h, w)

        # RSD kernel (falloff term is ignored)
        dgrid = dgrid.reshape(1, len(zdim), self.h, self.w)     # (1, d, h, w)
        omegas = omegas.reshape(-1, 1, 1, 1)                    # (o, 1, 1, 1)
        rsd = np.exp(1j * omegas / 3e8 * dgrid) / dgrid         # (o, d, h, w)
        return rsd, tgrid

    def forward(self, x, sqrt=True):
        raise NotImplementedError("RSD forward pass not implemented")



class RSD(RSDBase):

    def __init__(self, **kwargs):
        super(RSD, self).__init__(**kwargs)

    def forward(self, x, sqrt=True):
        """
        Args:
            x (float tensor, (bs, c, t, h, w)): input time-domain features.
            sqrt (bool): if True, take the square root before normalization.

        Returns:
            x (float tensor, (bs, c, d, h, w)): output space-domain features.
        """
        bs, c, t, h, w = x.shape
        assert t == self.t, \
            "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, \
            "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, \
            "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, \
            "feature dimension should be {:d}, got {:d}".format(self.in_plane, c)

        # propagate each feature dimension independently
        tdata = x.flatten(0, 1)                         # (bs*c, t, h, w)

        ## Step 1: convert measurement into FDH
        fdata = fft.rfft(tdata, dim=1)                  # (bs*c, t//2+1, h, w)
        fdata = fdata[:, self.freq_idx]                 # (bs*c, o, h, w)

        ## Step 2: define source phasor field
        phasor = self.coefs * fdata                     # (bs*c, o, h, w)
        phasor = F.pad(phasor, (0, w, 0, h))            # (bs*c, o, h*2, w*2)
        fsrc = fft.fftn(phasor, s=[-1, -1])             # (bs*c, o, h*2, w*2)

        ## Step 3: RSD propagation
        # WARNING: PyTorch is buggy when distributing complex tensors
        # here is a temporary workaround
        frsd, phase = self.frsd, self.phase
        if frsd.dim() == 5:
            frsd = torch.complex(frsd[..., 0], frsd[..., 1])
        if phase.dim() == 5:
            phase = torch.complex(phase[..., 0], phase[..., 1])
        fdst = fsrc.unsqueeze(2) * frsd
        # fdst = fsrc.unsqueeze(2) * self.frsd
        fdst = phase * fdst
        # fdst = self.phase * fdst                        # (bs*c, o, d, h*2, w*2)
        fvol = torch.sum(fdst, 1)                       # (bs*c, d, h*2, w*2)
        tvol = fft.ifftn(fvol, s=[-1, -1])
        tvol = tvol[:, :, h//2:h + h//2, w//2:w + w//2] # (bs*c, d, h, w)
        
        ## Step 4: post-process data
        tvol = torch.abs(tvol)                          # (bs*c, d, h, w)
        if not sqrt:
            tvol = tvol ** 2

        x = tvol.reshape(bs, c, self.d, h, w)
        x = self.actv(self.norm(x))
        return x
    
    

class RSDEfficient(RSDBase):
    """
    NOTE: this implementation does not zero-pad RSD kernel for efficiency.
    This results in sparser frequency sampling (4x memory saving) and 
    slightly noiser reconstruction results (with aliasing).
    """
    def __init__(self, **kwargs):
        super(RSDEfficient, self).__init__(efficient=True, **kwargs)

    def forward(self, x, sqrt=True):
        """
        Args:
            x (float tensor, (bs, c, t, h, w)): input time-domain features.
            sqrt (bool): if True, take the square root before normalization.

        Returns:
            x (float tensor, (bs, c, d, h, w)): output space-domain features.
        """
        bs, c, t, h, w = x.shape
        assert t == self.t, \
            "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, \
            "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, \
            "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, \
            "feature dimension should be {:d}, got {:d}".format(self.in_plane, c)

        # propagate each feature dimension independently
        tdata = x.flatten(0, 1)                         # (bs*c, t, h, w)

        ## Step 1: convert measurement into FDH
        fdata = fft.rfft(tdata, dim=1)                  # (bs*c, t//2+1, h, w)
        fdata = fdata[:, self.freq_idx]                 # (bs*c, o, h, w)

        ## Step 2: define source phasor field
        phasor = self.coefs * fdata                     # (bs*c, o, h, w)
        fsrc = fft.fftn(phasor, s=[-1, -1])             # (bs*c, o, h, w)

        ## Step 3: RSD propagation
        # WARNING: PyTorch is buggy when distributing complex tensors
        # here is a temporary workaround
        frsd, phase = self.frsd, self.phase
        if frsd.dim() == 5:
            frsd = torch.complex(frsd[..., 0], frsd[..., 1])
        if phase.dim() == 5:
            phase = torch.complex(phase[..., 0], phase[..., 1])
        fdst = fsrc.unsqueeze(2) * frsd
        # fdst = fsrc.unsqueeze(2) * self.frsd
        fdst = phase * fdst                             # (bs*c, o, d, h, w)
        # fdst = self.phase * fdst                        # (bs*c, o, d, h, w)
        fvol = torch.sum(fdst, 1)                       # (bs*c, d, h, w)
        tvol = fft.ifftn(fvol, s=[-1, -1])
        tvol = fft.ifftshift(tvol, dim=(-2, -1))
        
        ## Step 4: post-process data
        tvol = torch.abs(tvol)                          # (bs*c, d, h, w)
        if not sqrt:
            tvol = tvol ** 2

        x = tvol.reshape(bs, c, self.d, h, w)
        x = self.actv(self.norm(x))
        return x

