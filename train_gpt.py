import math
from dataclasses import dataclasses
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__() # call parent constructor. call the __init__ method of the parent class so I can use the methods and attributes of the parent class.
        assert config.n_embd % config.n_head == 0 # check if n_embd is divisible by n_head
        # key, query, value projections for all heads, but in batch
        # nn.Linear is a linear layer that applies a linear transformation to the incoming data: y = Ax + b

        # if n_embd = 768, then input shape is (batch_size, seq_len, 768)
        # and output shape is (batch_size, seq_len, 3 * 768)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # linear layer to project input to key, query, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # linear layer to project output to n_embd

        # regularization
        self.n_head = config.n_head # number of heads
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # create a lower triangular matrix of ones

    def forward(self, x):
        B, T, C = x.size() # get batch size, sequence length, and number of channels(embedding dimensinality n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        qkv = self.c_attn(x)
        # split last dim of size 3 * n_embd in to 3 parts each of size n_embd
        # now you get q, k, v of shape (B, T, n_embd)
        q,k,v = qkv.split(self.n_embd, dim=2) # split the output of c_attn into query, key, value

        # split n_embd into multiple heads
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, n_head, T, n_emb//n_head) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        # apply attention on all the projected vectors in batch
        # attention (materializes the large (T,T) matrix for all queries and keys)

        # k.transpose(-2, -1) - transpose the last two dimensions of k(swaps the last two dimensions of k)
        # k.size(-1) - get the size of the last dimension of k which is hs
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T) # scaled dot product attention

        # below is the masking step which makes sure that the attention is only on the left side of the current token
        # this is done by setting the upper triangular part of the attention matrix to -inf
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))

        # applies softmax function along the last dimension of att. This converts the attention scores to probabilities
        # that sum to 1 across each row
        att = F.softmax(att, dim = -1) # apply softmax to the attention matrix

        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs) # apply attention to values

        y = y.transpose(1,2).contiguous().view(B, T, C) # (B, T, nh, hs) -> (B, T, n_embd) # concatenate heads and put back into the original shape

        # output projection
        y = self.c_proj(y)

        return y # return the output of the attention layer


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # linear layer to project input to 4 * n_embd
        self.gelu = nn.GELU(approximate = 'tanh') # activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # nn.LayerNorm is a layer normalization layer that normalizes the input across the last dimension
        # this is done by subtracting the mean and dividing by the standard deviation
        self.ln1 = nn.LayerNorm(config.n_embd)    
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# decorator to automatically generate special methods like __init__() and __repr__()
@dataclass
class GPTConfig:
    block_size: int = 128 # size of the context window/max sequence length
    vocab_size: int = 50257 # size of the vocabulary/ number of tokens  50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # size of the embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word embedding layer
            # below creates a positional embedding layer that generates positional embedding
            # for each position in the sequence. With each position represented by vector of size config.n_embd
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding layer
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # layer normalization layer
        )) # create a dictionary to hold the transformer blocks

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # linear layer to project output to vocab size
