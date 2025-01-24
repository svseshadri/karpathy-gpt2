from dataclasses import dataclassimport torch
import torch.nn as nn
from torch.nn import functional as F

#------------------------------------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
"""Multi-headed self attention (!): parallel streams of self-attention, where outputs are concatenated.
 Each token outputs 3 vectors, query, key, and values.  First, query and key are multiplied together to 
  get the attention magnitude (how interesting is this token).   """
    def __init__(sef, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(congfig.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_heads=12, hs=64, so C = 768 channels in the Transformer
        # thus, nh is a batch dimension as follows
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        # autoregressive mask that ensure that we only tend to previous tokens, not future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # softmax normalizes attention to sum to one
        att = F.softmax(att, dim=-1)
        # attention matmul with values is a way to calculate a weighted sum of interesting tokens
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
"""Using the tanh approximation of the GELU activation to adhere to the GPT-2 paper, though an
exact GELU function now exists"""

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu   = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
"""Instantiates the block that h recurses over.  These are GPT2's hidden layers.
"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 384 # embedding dimension 

class GPT(nn.module):
"""Initializze an nn.moduleDict, in which submodules can be indexed with keys just like a dictionary.
wte = weights of token embeddings as nn.Embeddings which is just an array of numbers (indexed with stringS)
wpe = weights of positional embeddings
h = hidden layers, indexed using integers where the integer is the hidden layer number
ln_f = final layernorm according to the GPT-2 paper
lm_h = final classifier/language model head, projects from embd_dims (768) to the vocab size (50257)"""


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(        
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from HF"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), #124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), #350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        #init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #same, just the masked
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # these transposes are necessary because the original implementation was done in tensorflow and
        # the shape of these tensors isn't correct for pytorch format.  basically the openai checkpoints
        # use a "Conv1D" module, but we only want to use a vanilla module
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                #vanilla copy over the other parameters 
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ---------------------------------------------------------------------------------------------------------------
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")
