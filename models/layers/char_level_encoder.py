import torch.nn as nn


class CharLevelEncoder(nn.Module):

    def __init__(self, char_emb_size, max_w_len, kernel_size=5, padding=2, keep_prob=1.0):
        super(CharLevelEncoder, self).__init__()
        self.char_emb_size = char_emb_size
        self.max_w_len = max_w_len

        self.drop = nn.Dropout(1. - keep_prob)
        self.conv = nn.Conv1d(self.char_emb_size, self.char_emb_size, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(self.char_emb_size // 16, self.char_emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(self.max_w_len)

    def forward(self, x):
        x = x.view(-1, self.char_emb_size, self.max_w_len)
        x = self.drop(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
