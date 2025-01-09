import torch
import numpy as np
import scipy
import itertools

### Gerador de Ruído Correlacionado
def CorrNoiseGer(rho,m,n):
    Rho = torch.eye(n)
    for row in range(n):
        for col in range(n):
            if row != col:
                Rho[row][col]= rho
    C = torch.linalg.cholesky(Rho)
    gaussian_noise = torch.randn(m,n)
    return gaussian_noise@C
###
import itertools
def re_ref(y,channels,method):
    # Time Domain
    dimm = len(y.size())
    if method == 'AVG':
            y_avg = y[:,channels]
            y_out = y[:,channels] - y_avg.mean(dim=1).unsqueeze(1)
    if method == 'BD':
        bd = list(itertools.combinations(channels, 2))
        y_out = y[:,[i for i,j in bd]] - y[:,[j for i,j in bd]]
    if method == 'CAR':
        y_out = y[:,channels]-y.mean(dim=1).unsqueeze(1)
    if method == 'MB':
        if len(channels)==1:
            y_out = y[:,channels]
        else:
            bd = list(itertools.combinations(channels, 2))
            d = y[:,[i for i,j in bd]] - y[:,[j for i,j in bd]]
            y_out = torch.cat((y[:,channels], d), dim=1) 
    return y_out

#####


