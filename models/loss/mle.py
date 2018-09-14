import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()

    def forward(self, pred_params, tgts):
        cum_loss = 0
        # mu, s = pred_params[:,0], pred_params[:,1]
        # pred = torch.distributions.LogNormal(mu, s.exp())
        # tte, is_alive = tgts[:,0], tgts[:,1]
        # print('is_alive', is_alive)
        # print('tte', tte)
        # print('pred', pred)

        # print('log cdf', (1 - pred.cdf(tte) + 1e-5).log())
        # print('log prob', pred.log_prob(tte + 1e-5))
        # cum_loss += -(is_alive * (1 - pred.cdf(tte) + 1e-5).log())
        # cum_loss += -(1 - is_alive) * pred.log_prob(tte + 1e-5)
        # cum_loss = -(is_alive * (1 - pred.cdf(tte) + 1e-5).log() + (1 - is_alive) * pred.log_prob(tte + 1e-5))
        for pred_param, tgt in zip(pred_params, tgts):
            mu, s = pred_param[0], pred_param[1]
            pred = torch.distributions.LogNormal(mu, s.exp())
            # pdb.set_trace()
            # print('tgt', tgt.shape, type(tgt))
            tte, is_alive = tgt[0], tgt[1]
            tte = tte.cuda()

            print(type(tte), type(is_alive), tte.shape, is_alive.shape)
            print(is_alive)
            print(tte)
            print(mu, s)

            #pdb.set_trace()
            alive_loss = -((1 - pred.cdf(tte) + 1e-5).log())
            dead_loss = - pred.log_prob(tte + 1e-5)
            incr_loss = is_alive * alive_loss + (1 - is_alive) * dead_loss
            
            #if is_alive:
            #    incr_loss = -((1 - pred.cdf(tte) + 1e-5).log())
            #else:
                #print(f'dead/tte {tte}/mu {mu}, s.exp() {s.exp()}')
                #print(f'log prob', pred.log_prob(tte + 1e-5))
            #    incr_loss = -(1 - is_alive) * pred.log_prob(tte + 1e-5)

            # Debugging numerical instability
            if torch.isnan(incr_loss) or incr_loss == float('inf'):
                print("!!!!ERROR, tgts", tgts)
                if is_alive:
                    print("nan alive; pred cdf", pred.cdf(tte), "; pred", pred, "; log inner", 1 - pred.cdf(tte) + 1e-5, "; log", (1 - pred.cdf(tte) + 1e-5).log())
                else:
                    print("nan dead; pred log_prob", pred.log_prob(tte + 1e-5), "; tte + eps", tte + 1e-5, "; mu s.exp()", mu, s.exp())
                print("pred params: mu, s", mu, s)
                pdb.set_trace()

            cum_loss += incr_loss
    
        return cum_loss / tgts.shape[0]



