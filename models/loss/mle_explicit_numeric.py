import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()

        self.log_base = np.e # 16

    def smooth(self, N, D):
        OFFX = 1000
        smooth = Variable((F.relu(torch.max(N - OFFX * D, 0*(D - OFFX * N))) / (OFFX - 1.)).data)
        return smooth

    def log_density(self, mu, s2, tte):
        log_haz = -torch.log(s2 + 1e-5)
        numer = torch.pow(tte.log() / np.log(self.log_base) - mu, 2)
        smoothie = self.smooth(numer, s2)
        log_sur = - (numer + smoothie) / (s2 + smoothie)
        return log_haz + log_sur

    def log_tailmass(self, mu, s2, tte):
        numer = (tte.log() / np.log(self.log_base) - mu)
        denom = torch.sqrt(2 * s2)
        return torch.log(0.5 - 0.5*torch.erf((numer) / (denom)) + 1e-5)

    def forward(self, pred_params, tgts):
        cum_loss = 0
        for pred_param, tgt in zip(pred_params, tgts):
            mu, s = pred_param[0], pred_param[1]
            tte, is_alive = tgt[0], tgt[1]
            s2 = s.exp()

            if is_alive:
                loss = self.log_tailmass(mu, s2, tte)
            else:
                loss = self.log_density(mu, s2, tte)

            print("loss is:", loss, "is_alive is:", is_alive)
            if torch.isnan(loss) or loss == float('inf'):
                print("still getting inf with is_alive ==", is_alive)
            cum_loss += loss
        
        return -cum_loss
        # score_dead = self.log_density(mu, s2, time)
        # score_alive = self.log_tailmass(mu, s2, time)
        # losses =  - (mask * (score_dead * (1-censor) + score_alive * censor))





