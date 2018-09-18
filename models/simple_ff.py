import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, data_dir, D_in, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.D_in = D_in
        self.num_demographics = 2
        self.vocab_size = D_in - self.num_demographics
        self.embedding_dim = 128
        # self.model.apply(self.init_weights)

        self.model = nn.Sequential(
            # nn.Linear(D_in, 16),
            nn.Linear(self.num_demographics + self.embedding_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed.cuda()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, src):
        src_dem = src[:, :self.num_demographics]
        src_codes = src[:, self.num_demographics:].long()
        embedded_codes = [self.embed(torch.nonzero(c)[:,0]) for c in src_codes]
        embedded_codes = [e.mean(dim=0).unsqueeze(0) for e in embedded_codes]
        embedded_codes = torch.cat(embedded_codes)
        src = torch.cat((src_dem, embedded_codes), dim=1)
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
