import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()
        self.log_base = torch.FloatTensor([np.e]).cuda()
        self.age_max = torch.FloatTensor([120.0]).cuda()
        self.eps = 1e-7


    def log_intervalmass(self, mu, sigma2, time, max_ttd):
        return torch.log(self.cdf(mu, sigma2, max_ttd) - self.cdf(mu, sigma2, time) + self.eps)

    def debug(self, mu, sigma2, time, max_ttd):
        return torch.log(self.cdf_debug(mu, sigma2, max_ttd) - self.cdf_debug(mu, sigma2, time) + self.eps)

    def cdf(self, mu, sigma2, time):
        numer = (torch.log(time) - mu)
        denom = torch.sqrt(2 * sigma2)
        return 0.5 + 0.5*torch.erf(numer / denom)

    def cdf_debug(self, mu, sigma2, time):
        pdb.set_trace()
        numer = (torch.log(time) - mu)
        denom = torch.sqrt(2 * sigma2)
        return 0.5 + 0.5*torch.erf(numer / denom)

    def forward(self, pred_params, tgts, ages, use_intvl):
        cum_loss = 0

        for pred_param, tgt, age in zip(pred_params, tgts, ages):
            mu, s = pred_param[0], pred_param[1]
            pred = torch.distributions.LogNormal(mu, s.exp())
            tte, is_alive = tgt[0], tgt[1]
            tte = tte.cuda()

            if use_intvl:
                age = age.cuda()
                max_tte = self.age_max - age
                incr_loss = -self.log_intervalmass(mu, s.exp(), tte, max_tte)
                if is_alive and incr_loss < -self.eps and int(incr_loss) != 0:
                    print('Error: negative loss when patient is alive.', incr_loss)
                    #pdb.set_trace()
                    incr_loss = self.debug(mu, s.exp(), tte, max_tte)

            else:
                alive_loss = -((1 - pred.cdf(tte) + self.eps).log())
                dead_loss = - pred.log_prob(tte + self.eps)
                incr_loss = is_alive * alive_loss + (1 - is_alive) * dead_loss

                # Debugging numerical instability
                if torch.isnan(incr_loss) or incr_loss == float('inf'):
                    print("!!!!ERROR, tgts", tgts)
                    if is_alive:
                        print("nan alive; pred cdf", pred.cdf(tte), "; pred", pred, "; log inner", 1 - pred.cdf(tte) + self.eps, "; log", (1 - pred.cdf(tte) + self.eps).log())
                    else:
                        print("nan dead; pred log_prob", pred.log_prob(tte + self.eps), "; tte + eps", tte + self.eps, "; mu s.exp()", mu, s.exp())
                    print("pred params: mu, s", mu, s)
                    pdb.set_trace()

            cum_loss += incr_loss
    
        return cum_loss / tgts.shape[0]



