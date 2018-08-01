import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, data_dir, D_in, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.model = nn.Sequential(
            nn.Linear(D_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        # self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, src):
        pred = self.model(src)
        return pred
