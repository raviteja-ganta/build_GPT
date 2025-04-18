import os
import math
import time
import inspect
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all candidate parameters
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter out the parameters that do not require gradients
        
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no
        # i.e all weight tensors in matmuls + embeddings will be weight decayed, all biases and layernorms will not
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2] # get the parameters that are 2D
        no_decay_params = [p for n,p in param_dict.items() if p.dim()<2] # get the parameters that are not 2D

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available

        # The AdamW optimizer is an extension of the Adam optimizer that incorporates weight decay regularization.
        # The Adam optimizer is an adaptive learning rate optimization algorithm that is widely used in deep learning.  

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) # create an AdamW optimizer with the above parameters

        return optimizer # return the optimizer

# import the required libraries

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2") # get the encoding for gpt2
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) # convert to tensor

        if master_process:
            print(f"loaded {len(self.tokens)} tokens")  

        # state
        self.current_position = self.B * self.T * self.process_rank # current position in the tokens tensor
        # this is the position of the first token in the current batch
        # this is used to load the data in a distributed manner
        # so that each process gets a different batch of data

    def next_batch(self):
        # get the next batch of data
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # get the next B * T + 1 tokens. get the next batch of data
        x = (buf[:-1].view(B, T)) # (B, T) # create a tensor of shape (B, T) with the tokens
        y = (buf[1:].view(B, T)) # (B, T) # create a tensor of shape (B, T) with the tokens shifted by 1
        # advance the positon of the tensor
        self.current_position += B * T * self.num_processes # move the current position by B * T * num_processes
        # this is done to ensure after we split data in to num_processes gpu's, we want
        # to move by B * T * num_processes to get the next batch of data

        # if loading the next batch would be out of bounds, reset
        if self.current_position + B * T  * self.num_processes > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank 
        return x, y

# DDP launch for e.g with 8 gpus
# torchrun --standalone --nproc_per_node=8 train_gpt.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE

ddp = int(os.environ.get('RANK', -1)) != -1 # check if we are using DDP
if ddp:
    # use of DDT atm remands CUDA, we set the device appropriately according to the rank

    assert torch.cuda.is_available(), "DDP requires CUDA" # check if CUDA is available
    init_process_group(backend='nccl') # initialize the process group for DDP

    ddp_rank = int(os.environ['RANK']) # get the rank from the env variable # rank of each GPU
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # get the local rank from the env variable # this is local rank of gpu
    ddp_world_size = int(os.environ['WORLD_SIZE']) # get the world size from the env variable # this will be 8 for 8 GPUs

    device = f'cuda:{ddp_local_rank}' # set the device to the local rank
    torch.cuda.set_device(device) 

    master_process = ddp_rank == 0 # gpu with ddp_rank = 0, we are making this gpu as master_process for logging, checkpointing etc
else:
    # if not using DDP, just use the first GPU
    # vanilla, non-DDP training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True # we are using the first GPU as master process

    # attempt to auto detect the device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and hasattr(torch.backends, 'mps'):
        device = 'mps'
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    # set the seed for all GPUs
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # this is batchsize used in the paper which is actual number of tokens
# so 524288/1024 = 512 is the number of batches
# we cannot use this big batch size if we are using a single GPU
# so we can use gradient accumulation to simulate a larger batch size
# gradient accumulation is a technique used in deep learning to update the model weights after accumulating gradients over multiple mini-batches
# instead of updating the weights after each mini-batch, we accumulate the gradients over multiple mini-batches and then update the weights
# this allows us to use a larger effective batch size without increasing the memory requirements
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size" # check if total batch size is divisible by B * T
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # calculate the number of gradient accumulation steps
# The gradient accumulation steps is the number of times we will accumulate gradients before updating the model parameters.
# This is useful when we want to simulate a larger batch size than what can fit in memory.
# For example, if we have a total batch size of 1024 and a micro batch size of 16, we will have 64 gradient accumulation steps.
# This means that we will accumulate gradients for 64 batches before updating the model parameters.
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=>calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size) # create a data loader with batch size B and sequence length T

# below line speeds up training as we are using tensor float32
torch.set_float32_matmul_precision('high') # The argument 'high' indicates that you want to use a high precision level for matrix multiplication operations.
# Higher precision levels generally result in more accurate results but may be slower compared to lower precision levels.
model = GPT(GPTConfig(vocab_size = 50304)) # create a model object with the config object

model.to(device) # move the model to GPU
model = torch.compile(model) # compile the model for faster training

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model in DDP

raw_model = model.module if ddp else model # get the raw model from the DDP wrapper

# create learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    # if it > max_steps, return min_lr
    if it > max_steps:
        return min_lr
    
    # in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) # return the learning rate
    

# optimize the model
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # configure the optimizer for the model parameters

for step in range(max_steps):
    t0 = time.time() # get the current time
    optimizer.zero_grad() # zero the gradients

    loss_accum = 0.0 # initialize the loss accumulator
    for micro_step in range(grad_accum_steps): # iterate over the gradient accumulation steps
        x, y = train_loader.next_batch() # get the next batch of data
        x = x.to(device) # move the input to GPU
        y = y.to(device) # move the target to GPU

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y) # forward the model and get the logits and loss

        # The loss is divided by the number of gradient accumulation steps to ensure that the gradients are averaged over all steps.
        # we have to scale the loss to account for gradient accumulation
        # because gradients just add on each successive backward pass
        # addition of gradients corresponds to a SUM in objective function
        # instead of SUM we want MEAN. So we divide by the number of gradient accumulation steps
        loss = loss / grad_accum_steps # scale the loss by the number of gradient accumulation steps
        loss_accum += loss.detach() # accumulate the loss

        # backward pass
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # sync the gradients only on the last micro step
        loss.backward() # calculate the gradients

    if ddp:
        # reduce the loss across all GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # reduce the loss across all GPUs
        # this is done to ensure that the loss is averaged across all GPUs
        # and we want to ensure that the loss is averaged across all GPUs

    # clip the gradients to prevent exploding gradients
    # The function torch.nn.utils.clip_grad_norm_ iterates over the model parameters, computes the norm of the gradients, 
    # and clips the gradients if their norm exceeds the specified threshold (in this case, 1.0).
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the gradients

    # determine and set learning rate for this iteration
    lr = get_lr(step) # get the learning rate
    for param_group in optimizer.param_groups: # iterate over the optimizer parameter groups
        param_group['lr'] = lr # set the learning rate for the parameter group

    optimizer.step() # update the weights
    t1 = time.time() # get the current time

    dt = (t1 - t0) # calculate the time taken for the forward pass
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # calculate the number of tokens processed
    tokens_per_sec = (tokens_processed)/dt # calculate the tokens per second
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    # destroy the process group
    destroy_process_group() # destroy the process group for DDP
    # this is done to free up the resources used by DDP

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









  

