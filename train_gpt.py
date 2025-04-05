import math
from dataclasses import dataclass
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
    

    def forward(self, idx):

        # idx is of shape (B, T)
        B, T = idx.size() # get batch size and sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}" # check if sequence length is less than or equal to block size

        # forward the token and position embeddings
        # creates a tensor representing positions of each token in the sequence
        # with values ranging from 0 to T-1
        # ensures that the tensor is on the same device as idx
        # pos - resulting tensor which contains positional indices for each token in the sequence
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # create a tensor of shape (T) with values from 0 to T-1

        pos_emb = self.transformer.wpe(pos) # get the positional embeddings for the positions of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # get the token embeddings for the input tokens of shape (B, T, n_embd)

        x = tok_emb + pos_emb # add the token and positional embeddings together

        # forward the transformer blocks

        for block in self.transformer.h: # iterate over the transformer blocks
            x = block(x)

        # forward the final layer normalization and classfier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits # return the logits of the model

    @classmethod

    def from_pretrained(cls, model_type):
        # Loads pretrained GPT-2 model weights from huggingface to load in to our model

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print("Loading pretrained model from huggingface : %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_embd=768, n_head=12), # 124 M params
            "gpt2-medium": dict(n_layer=24, n_embd=1024, n_head=16), # 350 M params
            "gpt2-large": dict(n_layer=36, n_embd=1280, n_head=20), # 774 M params
            "gpt2-xl": dict(n_layer=48, n_embd=1600, n_head=25) # 1558 M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024

        # create a from scratch initialzed minGPT model
        config = GPTConfig(**config_args) # create a config object with the above args

        model = GPT(config) # create a model object with the config object

        sd = model.state_dict() # get the state dict of the model
        sd_keys = sd.keys() # get the keys of the state dict

        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask/buffer, not a param

        # init a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # load the huggingface model
        sd_hf = model_hf.state_dict()

        # copy the weights from the huggingface model to our model
        sd_keys_hf = sd_hf.keys() # get the keys of the huggingface model

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # ignore these, just a buffer

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # list of keys to transpose

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
         # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
 
        return model

# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

device = 'cpu'

# get a batch
import tiktoken
enc = tiktoken.get_encoding("gpt2") # get the encoding for gpt2

with open('input.txt', 'r') as f:
    text = f.read() # read the input text file
text = text[:1000] # take the first 1000 characters of the text
# prefix tokens

tokens = enc.encode(text) # encode the input text
B = 4
T = 32
buf = torch.tensor(tokens[:B*T+1]) # take the first B*T+1 tokens
x = buf[:-1].view(B, T) # (B, T) # create a tensor of shape (B, T) with the tokens
y = buf[1:].view(B, T) # (B, T) # create a tensor of shape (B, T) with the tokens shifted by 1

model = GPT(GPTConfig()) # create a model object with the config object

model.to(device) # move the model to GPU

logits = model(x) # forward the model (B, T) -> (B, T, vocab_size)
print(logits.size()) # print the size of the logits
import sys; sys.exit(0) # exit the program

model.eval()
num_return_sequences = 5    
max_length = 30

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
 # set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length: # while the sequence length is less than the maximum length
    # forward the model to get logits
    with torch.no_grad(): # no need to calculate gradients
        logits = model(x) # forward the model (B, T) -> (B, T, vocab_size)

        # take the logits at last position/token
        logits = logits[:, -1, :] # (B, vocab_size)

        # get the probabilities dim = -1 indicates the last dimension
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution
        # top_k sampling. do top-k sampling of 50
        # top_k probs here comes (5, 50), top_k indices (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)

        # sample from the top-k distribution
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)

        # gather the corresponding indices
        xcol = torch.gather(topk_indices, dim=-1, index=ix) # (B, 1)

        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated sequences
for i in range(num_return_sequences):
    tokens = x[i, : max_length].tolist() # get the tokens for the i-th sequence
    decoded = enc.decode(tokens) # decode the tokens to text
    print('>', decoded)







  

