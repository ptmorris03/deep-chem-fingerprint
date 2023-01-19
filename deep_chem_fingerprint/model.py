import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sin_pos_emb_weight(max_length, dims):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dims)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dims)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(max_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table


class PositionEmbedding(nn.Module):
    def __init__(self, dims: int, max_length: int):
        super().__init__()

        pos_emb = sin_pos_emb_weight(max_length, dims)
        self.pos_emb = nn.Parameter(torch.tensor(pos_emb, dtype=torch.float32))
        
    def forward(self, x):
        return x + self.pos_emb


class Mlp2(nn.Module):
    def __init__(self, dims: int):
        super().__init__()

        self.norm = nn.LayerNorm(dims)
        self.layer1 = nn.Linear(dims, dims * 4)
        self.layer2 = nn.Linear(dims * 4, dims)

    def forward(self, x):
        h = self.norm(x)
        h = self.layer1(h)
        h = F.gelu(h)
        return self.layer2(h)


class Attn(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()

        self.norm = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(dims, heads, batch_first=True)

    def forward(self, x, pad_mask=None):
        h = self.norm(x)
        return self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)


class ResBlock(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()

        self.mlp = Mlp2(dims)
        self.attn = Attn(dims, heads)

    def forward(self, x, pad_mask=None):
        h, _ = self.attn(x, pad_mask=pad_mask)
        x = x + h
        return x + self.mlp(x)


class Encoder(nn.Module):
    def __init__(
        self, 
        layers: int = 6, 
        dims: int = 256, 
        heads: int = 8, 
        vocab_size: int = 1000,
        max_length: int = 64,
        pad_token: int = 3
        ):
        super().__init__()

        self.pad_token = pad_token
        self.embed = nn.Embedding(vocab_size, dims, padding_idx=pad_token)
        self.pos = PositionEmbedding(dims, max_length)

        modules = [ResBlock(dims, heads) for i in range(layers)]
        self.encoder = nn.ModuleList(modules)

    def make_pad_mask(self, x):
        return x == self.pad_token

    def forward(self, x):
        pad_mask = self.make_pad_mask(x)

        h = self.embed(x)
        h = self.pos(h)

        for resblock in self.encoder:
            h = resblock(h, pad_mask=pad_mask)

        return h[:,0]


