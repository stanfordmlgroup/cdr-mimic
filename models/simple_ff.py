import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, data_dir, D_in, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.D_in = D_in
        self.model = nn.Sequential(
            nn.Linear(D_in, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        # self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, src):
        src = src.view(-1, self.D_in * 15)
        # b_size = src.size(0)
        pred = self.model(src)
        return pred

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {
            'D_in': self.D_in,
        }

        return model_args