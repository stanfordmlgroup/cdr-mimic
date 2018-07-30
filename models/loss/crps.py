import torch.nn as nn
import torch

class CRPSLoss(nn.Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, pred_params, tgts):
        loss_val = 0
        for pred_param, tgt in zip(pred_params, tgts):
            mu, s = pred_param[0], pred_param[1]
            pred = torch.distributions.LogNormal(mu, s.exp())
            tte, is_alive = tgt[0], tgt[1]
            # print("tte", tte)
            # print("log prob", pred.log_prob(tte + 1e-5))
            # print("cdf", pred.cdf(tte))
            loss_val += - ((1 - is_alive) * pred.log_prob(tte + 1e-5) + (1 - pred.cdf(tte) + 1e-5).log() * is_alive)
        # print("loss", loss_val)

        return loss_val / tgts.shape[0]
