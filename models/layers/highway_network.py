import torch.nn as nn


class HighwayNetwork(nn.Module):
    def __init__(self, num_layers, num_channels, dropout_prob):
        super(HighwayNetwork, self).__init__()

        layers = [HighwayLayer(num_channels, dropout_prob) for _ in num_layers]
        self.highway_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.highway_layers(x)

        return x


class HighwayLayer(nn.Module):
    def __init__(self, num_channels, dropout_prob):
        super(HighwayLayer, self).__init__()
        self.transform = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=1),
                                       nn.ReLU(inplace=True))
        self.gate = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=1),
                                  nn.Sigmoid())
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        h = self.transform(x)
        t = self.gate(x)
        x = t * h + (1 - t) * x
        x = self.drop(x)

        return x
