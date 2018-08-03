import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()

    def forward(self, pred_params, tgts):
        cum_loss = 0
        for pred_param, tgt in zip(pred_params, tgts):
            mu, s = pred_param[0], pred_param[1]
            pred = torch.distributions.LogNormal(mu, s.exp())
            # pred = self.arithmetic_mean(mu, s)
            # print("ARITHMETIC pred is:",pred)
            tte, is_alive = tgt[0], tgt[1]

            # Convert tte in sec to days
            # tte = tte / 86400
            # print("tte", tte)
            # print("log prob", pred.log_prob(tte + 1e-5))
            # print("cdf", pred.cdf(tte))

            if is_alive:
                incr_loss = -((1 - pred.cdf(tte) + 1e-5).log())
            else:
                incr_loss = -(1 - is_alive) * pred.log_prob(tte + 1e-5)
            if torch.isnan(incr_loss) or incr_loss == float('inf'):
                if is_alive:
                    print("nan alive; pred cdf", pred.cdf(tte), "; pred", pred, "; log inner", 1 - pred.cdf(tte) + 1e-5, "; log", (1 - pred.cdf(tte) + 1e-5).log())
                else:
                    print("nan dead; pred log_prob", pred.log_prob(tte + 1e-5), "; tte + eps", tte + 1e-5)
                print("pred params: mu, s", mu, s)
            cum_loss += -((1 - pred.cdf(tte) + 1e-5).log()) if is_alive else -(1 - is_alive) * pred.log_prob(tte + 1e-5)
            print("loss_val per", cum_loss)
        print("loss val in mle.py", cum_loss)

        return cum_loss / tgts.shape[0]



