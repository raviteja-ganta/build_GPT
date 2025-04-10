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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) # flash attention
        # (B, nh, T, hs) # scaled dot product attention
        # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs) # apply attention to values

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
    block_size: int = 1024 # size of the context window/max sequence length
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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # share the weights of the word embedding layer and the output layer

        # init params
        self.apply(self._init_weights) # initialize the weights of the model
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # initialize the weights of the linear layer with normal distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize the weights of the embedding layer with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

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

        loss = None
        if targets is not None: # targets shape is (B, T)
        # logits.view(-1, logits.size(-1)) - reshape logits to (B*T, vocab_size)
        # targets.view(-1) - reshape targets to (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # calculate the loss using cross entropy

        return logits, loss # return the logits and loss

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

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2") # get the encoding for gpt2
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) # convert to tensor
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        # get the next batch of data
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # get the next B * T + 1 tokens. get the next batch of data
        x = (buf[:-1].view(B, T)) # (B, T) # create a tensor of shape (B, T) with the tokens
        y = (buf[1:].view(B, T)) # (B, T) # create a tensor of shape (B, T) with the tokens shifted by 1
        # advance the positon of the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0
        return x, y

# attempt to autodetect the device
import time
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")
torch.manual_seed(1337)
if torch.cuda.is_available():
    # set the seed for all GPUs
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024) # create a data loader with batch size 4 and sequence length 128
# below line speeds up training as we are using tensor float32
torch.set_float32_matmul_precision('high') # The argument 'high' indicates that you want to use a high precision level for matrix multiplication operations.
# Higher precision levels generally result in more accurate results but may be slower compared to lower precision levels.
model = GPT(GPTConfig(vocab_size = 50304)) # create a model object with the config object

model.to(device) # move the model to GPU
model = torch.compile(model) # compile the model for faster training

# optimize the model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8) # create an optimizer for the model parameters
for i in range(50):
    t0 = time.time() # get the current time
    # basically we are getting 50 batches of data
    x, y = train_loader.next_batch() # get the next batch of data
    x, y = x.to(device), y.to(device) # move the data to GPU
    optimizer.zero_grad() # zero the gradients
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y) # forward the model
    loss.backward() # backward pass
    # clip the gradients to prevent exploding gradients
    # The function torch.nn.utils.clip_grad_norm_ iterates over the model parameters, computes the norm of the gradients, 
    # and clips the gradients if their norm exceeds the specified threshold (in this case, 1.0).
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the gradients
    optimizer.step() # update the weights
    t1 = time.time() # get the current time

    dt = (t1 - t0) # calculate the time taken for the forward pass
    tokens_processed = train_loader.B * train_loader.T # calculate the number of tokens processed
    tokens_per_sec = (train_loader.B * train_loader.T)/dt # calculate the tokens per second
    print(f"step {i:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0) # exit the program

model.eval()
num_return_sequences = 5    
max_length = 30

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)


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









  

