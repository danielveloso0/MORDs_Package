import torch 
import numpy as np
from scipy.stats import f
###
def aCSM(y,M,fs,alpha=None,cv_only=False):
    size,N = y.shape
    win_size = size//M
    y2 = y[:win_size*M,:].reshape(win_size,M,N)
    Y = fft(y2,dim=0)
    F = torch.linspace(0, fs / 2, win_size // 2+1)
    csm = torch.mean(torch.cos(torch.angle(Y)),dim=1)**2 + torch.mean(torch.sin(torch.angle(Y)),dim=1)**2
    aCSM = torch.mean(csm,dim=1)
    if alpha is not None:
        nRuns = int(1e5)
        y = torch.distributions.Chi2(df=2).sample((nRuns, N)) / (2 * M)
        cv = np.quantile(y.mean(dim=1).numpy(),1-alpha)
        if cv_only:
            return cv
        return aCSM, F, cv
    else:
        return aCSM, F
    

    #######

def pCSM(y,M,fs,alpha=None,cv_only=False):
    size,N = y.shape
    win_size = size//M
    y2 = y[:win_size*M,:].reshape(win_size,M,N)
    Y = fft(y2,dim=0)
    F = torch.linspace(0, fs / 2, win_size // 2+1)
    csm = torch.mean(torch.cos(torch.angle(Y)),dim=1)**2 + torch.mean(torch.sin(torch.angle(Y)),dim=1)**2
    pCSM = torch.prod(csm,dim=1)**(1/N)
    if alpha is not None:
        nRuns = int(1e5)
        y = torch.distributions.Chi2(df=2).sample((nRuns, N)) / (2 * M)
        y_geo = torch.prod(y,dim=1)**(1/N)
        cv = np.quantile(y_geo.numpy(),1-alpha)
        if cv_only:
            return cv
        return pCSM, F, cv
    else:
        return pCSM, F
        
####

def aMSC(y,M,fs,alpha=None,cv_only=False):
    size,N = y.shape
    win_size = size//M
    y2 = y[:win_size*M,:].reshape(win_size,M,N)
    Y = fft(y2,dim=0)
    F = torch.linspace(0, fs / 2, win_size // 2+1)
    msc = (torch.abs(Y.sum(dim=1))**2)/(M*torch.sum(torch.abs(Y)**2,dim=1))
    aMSC = torch.mean(msc,dim=1)
    if alpha is not None:
        nRuns = int(1e6)
        y = torch.distributions.Beta(1,(M-1)).sample((nRuns, N))
        cv = np.quantile(y.mean(dim=1).numpy(),1-alpha)
        if cv_only:
            return cv
        return aMSC, F, cv
    else:
        return aMSC, F
####

def pMSC(y,M,fs,alpha=None, cv_only=False):
    size,N = y.shape
    win_size = size//M
    y2 = y[:win_size*M,:].reshape(win_size,M,N)
    Y = fft(y2,dim=0)
    F = torch.linspace(0, fs / 2, win_size // 2+1)
    msc = (torch.abs(Y.sum(dim=1))**2)/(M*torch.sum(torch.abs(Y)**2,dim=1))
    aMSC = torch.prod(msc,dim=1)**(1/N)
    if alpha is not None:
        nRuns = int(1e6)
        y = torch.distributions.Beta(1,(M-1)).sample((nRuns, N))
        y_geo = torch.prod(y,dim=1)**(1/N)
        cv = np.quantile(y_geo.numpy(),1-alpha)
        if cv_only:
            return cv
        return aMSC, F, cv
    else:
        return aMSC, F
########

def aLFT(y,L,fs,fo,alpha=None,cv_only=False):
    size,N = y.shape
    nfft = round(size/2)
    pfo = round((fo*nfft)/(fs/2))
    Y = fft(y,dim=0)
    Y = Y[:nfft,:]
    Yfo = torch.abs(Y[pfo])**2
    Ydem = (1/L)*(torch.sum(torch.abs(Y[pfo-L//2:pfo,:])**2,dim=0) + torch.sum(torch.abs(Y[pfo+1:pfo+L//2+1,:])**2,dim=0))
    alft = torch.mean(Yfo/Ydem)
    if alpha is not None:
        nRuns = int(1e6)
        y = torch.distributions.FisherSnedecor(2,2*L).sample((nRuns, N))
        cv = np.quantile(y.mean(dim=1).numpy(),1-alpha)
        if cv_only:
            return cv
        return alft, cv
    else:
        return alft
    
####

def pLFT(y,L,fs,fo,alpha=None,cv_only=False):
    size,N = y.shape
    nfft = round(size/2)
    pfo = round((fo*nfft)/(fs/2))
    Y = fft(y,dim=0)
    Y = Y[:nfft,:]
    Yfo = torch.abs(Y[pfo])**2
    Ydem = (1/L)*(torch.sum(torch.abs(Y[pfo-L//2:pfo,:])**2,dim=0) + torch.sum(torch.abs(Y[pfo+1:pfo+L//2+1,:])**2,dim=0))
    plft = torch.prod(Yfo/Ydem)**(1/N)
    if alpha is not None:
        nRuns = int(1e6)
        y = torch.distributions.FisherSnedecor(2,2*L).sample((nRuns, N))
        y_geo = torch.prod(y,dim=1)**(1/N)
        cv = np.quantile(y_geo.numpy(),1-alpha)
        if cv_only:
            return cv
        return plft, cv
    else:
        return plft
####

def MLFT(y,L,fs,fo,alpha=None,cv_only=False):
    size,N = y.shape
    nfft = round(size/2)
    pfo = round((fo*nfft)/(fs/2))
    Y = fft(y,dim=0)
    Y = Y[:nfft,:]
    Yfo = torch.abs(Y[pfo])**2
    Ydem = (1/L)*(torch.sum(torch.abs(Y[pfo-L//2:pfo,:])**2,dim=0) + torch.sum(torch.abs(Y[pfo+1:pfo+L//2+1,:])**2,dim=0))
    mlft = torch.sum(Yfo)/torch.sum(Ydem)
    if alpha is not None:
        cv = f.ppf(1 - alpha, 2 * 5, 2 * 5 * L)
        if cv_only:
            return cv
        return mlft, cv
    else:
        return mlft
