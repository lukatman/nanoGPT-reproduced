from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # x of shape (B, T, C)
        # nn.linear only acts on the last dimension, so output is (B, T, C*3)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask -> to only allow current token to check itself and those before it
        # tril: lower-triangulates the matrix
        # .view(...): reshapes to [1, 1, block_size, block_size] (1, 1 = batch_size, n_head)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd
        # calculate query, key, value for all heads in batch and move head forward to be the batch dim
        # nh: "number of heads"; hs: "head size"; C = nh * hs, "no. of channels"
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # view() -> (B, T, nh, hs); transpose() -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # QK^T / sqrt(n_embd). k.transpose(-2, -1) gives (B, nh, hs, T)
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # > pytorch automatically considers the last 2 dims only for matrix multiplication
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # apply the bias buffer and fill upper triangle with -inf
        att = F.softmax(att, dim=-1)
        y = att @ v # weighted sum of the values of the tokens
                    # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connections
        # attention is an aggergation/pooling/weighted sum/reduction func
        x = x + self.mlp(self.ln_2(x))  # mlp = ffn, processes each token individually and maps
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024   # max context length
    vocab_size: int = 50257  # 50,000 BPE merges + 256 bytes tokens + 1 <eos>
    n_layer: int = 12        # no. of transformer blocks
    n_head: int = 12         # no. of attention heads (multi-head)
    n_embd: int = 768        # embedding dimension


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        '''
        Model sequence:
        1. token embedding
        2. + positional embedding
        3. passed into transformer blocks
        4. outputs of which are normalized
        5. linear + softmax -> output
        '''        
        # All randomly initialized by PyTorch by default
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),               # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),               # positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd)                                  
        ))
        # final classifier (n_embd -> vocab)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx is of shape (B, T)
        # it is the index of the tokens
        # the token we get by looking it up in the vocab table, wte
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # calls wpe to return position embeddings of rows from 0 to T-1
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # > wte and wpe already defined as embeddings, so they append n_embd as a dimension to the shapes of pos and idx
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # (T, vocab_size) -> each row is one position in the sequence (pos 0, pos 1, etc.)
        #                  > each col is an index for one token in the original vocab
        # so, highest scoring pos -> get column -> use col number as index into tokenizer vocabulary
        return logits



    
    @classmethod
    def from_pretrained(cls, model_type):
        '''Loads pretrained GPT-2 weights from HF'''
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257 # always for gpt model checkpoints
        config_args['block_size'] = 1024  # context length - always same for gpt model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args) # ** unpacks dict
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask

        # init a hf/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the params are aligned and match in names n' shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #ignore
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # we have to transpose these weights when we impose them
        # because openai checkpoints use Conv1D, but we just want to use regular Linear
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf) != {len(sd_keys)}}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment- need to transpose the conv1d weights
                assert sd_hf[k].shape[::-1] == sd[k].shape #[::-1] reverses shape
                with torch.no_grad(): # no need for gradient tracking
                    sd[k].copy_(sd_hf[k].t()) 
                    # copy_(): in-place operation in pytorch
            else:
                # vanilla copy over the other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model 
    
# ------------------------------------------------
num_return_sequences = 5
max_length = 30

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"using device: {device}")

model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8) - copy tokens 5 times
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position (-1 of T, i.e., sequence)
        logits = logits[:, -1, :] # (B, vocab_size)
        # get probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hf pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs (selection with respect to weights)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # maps back to vocab index
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist() # selects row i, up to max_length tokens
    decoded = enc.decode(tokens)
    print(">", decoded)