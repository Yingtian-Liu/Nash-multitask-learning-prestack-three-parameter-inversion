import torch
from torch.nn.functional import conv1d
from torch import nn
from bruges.filters import wavelets
import math
import numpy as np

class forward_model(nn.Module):
    def __init__(self, wavelet, resolution_ratio=6):
        super(forward_model, self).__init__()
        self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
        self.resolution_ratio = resolution_ratio
    def cuda(self):
        self.wavelet = self.wavelet.cuda()

    def forward(self, x):

        Vp = x[:,0,:]
        Vs = x[:,1,:]
        Den = x[:,2,:]
        Vp0 = torch.tensor(2865.6995)
        Vs0 = torch.tensor(1350.7539)
        Den0 = torch.tensor(2.2361467)

        k=0.25
        i=[5,10,15,20,25,30]
        i= np.round(np.deg2rad(i),4)

        nx, ny, nz = Vp.shape[0], len(i), Vp.shape[-1]
        Elastic_impedance = torch.zeros((nx, ny, nz)).to('cuda')
        angle = torch.tensor(i)
        
        for m in range(len(angle)):
            a= 1 + torch.tan(angle[m]) *torch.tan(angle[m])
            b= -8 * k * torch.sin(angle[m]) * torch.sin(angle[m])
            c= 1 - 4 * k * torch.sin(angle[m]) * torch.sin(angle[m])
            Elastic_impedance[:,m,:] = Vp0 * Den0 * (Vp / Vp0) ** a * \
                                  (Vs / Vs0) ** b * \
                                  (Den / Den0) ** c  
        rc = torch.zeros((Elastic_impedance.shape[0], Elastic_impedance.shape[1], Elastic_impedance.shape[2]))       
        rc1 = torch.zeros((Elastic_impedance.shape[0], Elastic_impedance.shape[1], Elastic_impedance.shape[2]-1))
        for m in range(Elastic_impedance.shape[1]):
            x_d = Elastic_impedance[:,m,1:] - Elastic_impedance[:,m,:-1]
            x_a = (Elastic_impedance[:,m,1:] + Elastic_impedance[:,m,:-1]) / 2
            rc1[:,m,:] = x_d / x_a 
        rc[:,:,1:] = rc1
        
        f=[40]
        wavelet, wavelet_time = wavelets.ricker(0.2, 1e-3, f) 
        wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    
        tmp_synth = torch.zeros_like(rc)
    
    
        for m in range(Elastic_impedance.shape[1]):
            rc1 = torch.tensor(rc[:,m,:]).unsqueeze(dim=1).float()
            tmp_synth1 = conv1d(rc1, wavelet, padding=int(wavelet.shape[-1] / 2))
            tmp_synth [:,m,:]=tmp_synth1.squeeze().float()

        tmp_synth = torch.tensor(tmp_synth).to('cuda')

    
        return tmp_synth
