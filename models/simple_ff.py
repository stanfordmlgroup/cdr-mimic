import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, data_dir, D_in, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.D_in = D_in
        self.num_demographics = 2
        self.vocab_size = D_in - self.num_demographics
        self.embedding_dim = 256
        # self.model.apply(self.init_weights)

        self.model = nn.Sequential(
            nn.Linear(D_in, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.embed = nn.Linear(self.vocab_size, self.embedding_dim)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, src):
        # src = src.view(-1, self.D_in)
        # b_size = src.size(0)
        src = src.float()
        src_dem = src[:self.num_demographics]
        src_codes = src[self.num_demographics:]
        src = torch.cat((src_dem,self.embed(src_codes)))
        pred = self.model(src)
        # embed = self.embed(src[self.num_demographics:])
        # return pred, embed
        return pred

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {
            'D_in': self.D_in,
        }

        return model_args
