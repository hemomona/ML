#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : train_gpt2.py
# Time       ：2024/6/13 0:57
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
ways to improve speed:
    1. down sample tensor precision, F32, TF32, BF16
    2. torch.compile(), kernel fusion
    3. flash-attention, kernel fusion, focus on reducing HBM trips
    4. fix the ugly number which is not the power of 2, 50257 to 50304
    5. gradually increase the batch size, initial training steps would generate very correlated weights, it is very rough
    and can only tell which tokens do appear and which do not, so there is no need to use a big data. It would be efficient
    and meaningful to use a big batch size after the model has learned simple things.  - skipped by Andrej
    6. torch.distributed,
ways to improve performance:
    1. gradient norm clipping, prevent the shock when encountering bad batches
    2. dynamic learning rate: cosine decay with linear warmup
    3. weight decay, sort of regularization, force the model to use more channels
    4. gradient accumulation, simulate in a serial way any arbitrary batch size
"""
import inspect
import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, channels (n_embd)
        # calculate q, k, v for all heads in a batch
        # e.g. GPT-2 (124M) has 768 channels = 12 (n_head) * 64 (head size)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # autoregressive mask makes sure the tokens only attend to tokens before them and never to tokens in the future
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # DIM = -1
        y = att @ v

        # the above 4 lines are written by Andrej, which can be replaced by FlashAttention
        # - a scheme includes more flops but is much more faster because of the avoidance of many flops related to HBM.
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True), however, is not available before PyTorch 2.0
        # and flash-attention compatibility with ninja and C++ on Windows sucks...

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # Gaussian Error Linear Units is better handling dead RELU neuron
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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # NANOGPT_SCALE_INIT
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)  # NANOGPT_SCALE_INIT

    def forward(self, x):
        # tokens communicate, aggregation function - pooling function - weighted sum function - reduce operation
        x = x + self.attn(self.ln_1(x))
        # happens at every single token individually, attn is to reduce and mlp is to map
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50000 byte pair encoding (BPE) + 256 + 1
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimensions


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(  # wrapped k-v dictionary module
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # hidden layer
            ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
        ))
        # final classifier - language model head - project from n_embd dimensions to vocab_size dimensions
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme mentioned in Attention Is All You Need
        # self.transformer.wte would be initialized twice in _init_weights, 1st in elif, 2nd in if
        self.transformer.wte.weight = self.lm_head.weight

        # init all params in self
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # usually we would init std as 1/sqrt(features),
            # but 0.02 comes from the original paper, which is a tradeoff in 4 types of gpt2.
            # without this scale, the std would increase along with the residual stream
            # there is attention layer and mlp layer, so n_layer needs to be timed by 2
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # zero the default bias which is uniform
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()  # batch size * time dimension (sequence length)
        assert T <= self.config.block_size, f"too long length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # position index uses same device with its idx
        pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and final classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            # flatten logits to 2D (B*T, vocab_size) and also targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from huggingface"""
        # assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        assert model_type in {'gpt2'}  # there is only gpt2 locally
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt2: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            # 'gpt2-medium':    dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            # 'gpt2-large':     dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            # 'gpt2-xl':        dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # init our GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init huggingface/transformers gpt2 model
        model_hf = GPT2LMHeadModel.from_pretrained('dataroot/models/openai-community/gpt2')
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # transpose these weights from tensorflow openai trained
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
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
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # we want to do weight decay for those participated in matrix multiplications and embeddings
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # including 1D tensors, layer norms, scales, biases
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version (kernel fusion) if it is available, which is NO in 1.13 Torch
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and 'cuda' in device
        # print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # self.current_position = 0
        self.current_position = self.B * self.T * self.process_rank  # for DDP

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # input
        y = buf[1:].view(B, T)  # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# # just load the pretrained weights from huggingface
# model = GPT.from_pretrained('gpt2')
# print('did not crash!')

# # load pretrained model or random model to generate text
# num_return_sequences = 5
# max_length = 30
# # model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig)
# model.eval()
# model.to('cuda')

# # get a data batch
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to('cuda')  # for tensors, .to(device) returns a pointer
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=4 train_gpt2.py

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='mpi')  # 
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"using device: {device}")

# create model
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 8
T = 128
assert total_batch_size % (B * T * ddp_world_size) == 0, \
    "ensure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:  # otherwise, each process on each GPU would print once
    print(f"total desired batch size: {total_batch_size}")
    print(f"calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
torch.set_float32_matmul_precision('high')  # use TF32 precision

model = GPT(GPTConfig(vocab_size=50304))  # use extra memory but actually speed up the process
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# torch.compile() can be faster, however, it is not available before PyTorch 2.0
# without torch.compile(), there are lots of traversing between GPU calculation and High Bandwidth Memory (HBM)
# with it - an example of kernel fusion - it can combine some operations and return at a single time.

# logits, loss = model(x, y)
# print(loss)  # tensor(10.9724, grad_fn=<NllLossBackward0>) == -ln(1/50527) == cross_entropy

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) min_lr after decay process
    if it > max_steps:
        return min_lr
    # 3) cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # this would cause OOM in my Ampere GPU
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # the loss must be scaled globally
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # so, until the last step of grad_accum_steps, the loss would not be synchronized.
        loss.backward()  # it is naturally accumulated

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # when training encounters a bad batch, it would generate a big loss, which consequently derive great gradients and
    # can shock the model, gradient norm clipping can prevent this
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # use cosine decay learning rate with linear warmup
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()  # the CPU just delegate jobs on GPU and then run sequentially
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    # same batch:   step49, loss: 0.00279676984064281
    # train_loader: step49, loss: 6.446619510650635
    # init std:     step49, loss: 6.799220561981201
    # (8, 128):     step49, loss: 7.0469770431518555, dt: 0.31s, tokens per sec: 3279.17
    # TF32:         step49, loss: 6.964852809906006, dt: 0.24s, tokens per sec: 4281.92
    # gradient clip:step  49, loss: 6.696885, norm: 2.116803, dt: 0.32s, tokens per sec: 3190.98
    # gradient accu:step  0, loss: 10.935869, norm: 25.713163, lr: 0.000060, dt: 121.57s, tokens per sec: 4312.73
    if master_process:
        print(f"step{step:3d}, loss: {loss_accum.item():6f}, norm: {norm:4f}, lr: {lr:6f}, "
              f"dt: {dt:.2f}s, tokens per sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

# # prefix tokens
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")  # 15496, 11, 314, 1101, 257, 3303, 2746, 11
# tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
# x = tokens.to('cuda')
#
# # generate answers, now (B, T) == (5, 8)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():  # no need to backward or cache tensors
#         logits = model(x)  # (B, T, vocab_size)
#         logits = logits[:, -1, :]  # last position
#         probs = F.softmax(logits, dim=-1)  # get the probability DIM = -1
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # do top 50 sampling, (5, 50), (5, 50) DIM = -1
#         ix = torch.multinomial(topk_probs, 1)  # select a token, (5, 1)
#         xcol = torch.gather(topk_indices, -1, ix)  # gather the corresponding indices
#         x = torch.cat((x, xcol), dim=1)
#
# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
