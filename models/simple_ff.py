import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, data_dir, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.model = nn.Sequential(
            nn.Linear(9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, src):
        pred = self.model(src)
        return pred
