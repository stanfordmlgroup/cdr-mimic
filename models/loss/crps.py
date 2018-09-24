import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class CRPS(nn.Module):
    def __init__(self):
        super(CRPS, self).__init__()
        self.K = 64
        self.log_base = torch.FloatTensor([np.e]).cuda()
        self.age_max = torch.FloatTensor([120.0]).cuda()
        
    def I_ln(self, mu, sigma2, y, g):
        s = torch.sqrt(sigma2)
        norm = torch.distributions.normal.Normal(mu, s)
        splits = y * torch.tensor(np.linspace(1e-8, 1, self.K).astype(np.float32))
        xdiffs = splits.narrow(-1, 1, 31) - splits.narrow(-1, 0, 31)
        Fs = (norm.cdf(splits.log() / torch.log(self.log_base)) * g(splits)).pow(2)
        Fdiffs = 0.5 * (Fs.narrow(-1, 1, 31) + Fs.narrow(-1, 0, 31))
        L = torch.sum(Fdiffs * xdiffs, -1, keepdim=True)
        return L

    def CRPS_surv_ln(self, mu, sigma2, time, censor, max_ttd, use_intvl):
        Y = time
        T = max_ttd
        I = lambda y: self.I_ln(mu, sigma2, y, lambda y_: 1)
        I_ = lambda y: self.I_ln(-mu, sigma2, 1/y, lambda y_: y_.pow(-1))

        if use_intvl:
            crps = I(Y) + I_(T) + (1 - censor) * (I_(Y) - I_(T))
        else:
            crps = I(Y) + (1 - censor) * I_(Y)
        return crps

    def I_norm(self, mu, sigma2, y):
        s = torch.sqrt(sigma2)
        ystd = (y - mu) / s
        norm = torch.distributions.normal.Normal(mu, s)
        norm2 = torch.distributions.normal.Normal(mu, s / float(np.sqrt(2.)))
        ncdf = norm.cdf(y)
        npdf = norm.log_prob(y).exp()
        n2cdf = norm2.cdf(y)
        return s * (ystd * ncdf.pow(2) + 2 * ncdf * npdf * s - float(1./np.sqrt(np.pi)) * n2cdf)

    def CRPS_surv_norm(self, mu, sigma2, time, censor, max_ttd, use_intvl):
        Y = time.log() / torch.log(self.log_base)
        T = max_ttd.log() / torch.log(self.log_base)
        I = lambda y: self.I_norm(mu, sigma2, y)
        I_ = lambda y: self.I_norm(-mu, sigma2, y)

        if use_intvl:
            crps = I(Y) + I_(-T) + (1 - censor) * (I_(-Y) - I_(-T))
        else:
            crps = I(Y) + (1 - censor) * I_(-Y)
        return crps

    def forward(self, pred_params, tgts, ages, use_intvl):

        cum_loss = 0

        for pred_param, tgt, age in zip(pred_params, tgts, ages):
            mu, s = pred_param[0], pred_param[1]
            pred = torch.distributions.LogNormal(mu, s.exp())
            tte, is_alive = tgt[0], tgt[1]
            tte = tte.cuda()
            age = age.cuda()
            max_tte = self.age_max - age
            incr_loss = self.CRPS_surv_norm(mu, s.exp(), tte, is_alive, max_tte, use_intvl)

            # Debugging numerical instability
            if torch.isnan(incr_loss) or incr_loss == float('inf'):
                print("!!!!ERROR, tgts", tgts)
                pdb.set_trace()
            cum_loss += incr_loss

        return cum_loss / tgts.shape[0]



