import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import os



###############################










def download_data(url = "https://www.gutenberg.org/files/1661/1661-0.txt", filename="input.txt"):
    """
    By default, use Bible corpus.
    """

    if not os.path.exists(filename):
        print(f"Downloading dataset from {url} to {filename}.")
        r = requests.get(url)
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(r.text)

    else:
        print("Dataset already exists. Skipping download.")


download_data()

text = open("input.txt", encoding = "utf-8").read()
print(text[4000:4250])
print("... excerpt from training data.")



# character-level tokenizer. TODO: Replace with BPE.

chars = sorted(list(set(text)) # vocab
vocab_size = len(chars) # v
print(f"vocab_size: {vocab_size}")

stoi = {ch:i for i, ch in enumerate(chars)) # str to int
itos = {i:ch for i, ch in enumerate(chars)) # int to str

def encode(s):
    return [stoi[c] for c in s]
def decode(s):
    return [itos[i] for i in s]

# encode the entire text:
data = torch.tensor(encode(text), dtype = torch.long)


class BibleText(Dataset):
    def __init__(self, data, block_size):
        super().__init__()
        self.data = data
        self.block_size = block_size

        # The quick brown fox >> encoded as chunks of [block_size]
        # 

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        dummy_x = self.data[idx:idx+block_size]
        dummy_y = self.data[idx+1:idx+block_size+1]
        return dummy_x, dummy_y


# hyperparameters
block_size = 128
batch_size = 1024

dataset = BibleText(data, block_size)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# model_definition

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_heads, block_size, dropout = 0.4):
        super().__init__()
        
        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads # h
        self.per_head_dim = embedding_dim // num_heads # d
        self.scale = self.head_dim ** (-0.5) # scaled dpa
        self.embedding_dim = embedding_dim 

        self.qkv = nn.Linear(embed_dim, embed_dim * 3) # expand and split. 
        self.proj = nn.Linear(embed_dim, embed_dim) # expression
        self.dropout = nn.Dropout(dropout)

        # masking for causal attention
        # mask the future tokens from the nodes in the graph
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        # register_buffer = part of module's state, but not learnable.
        # Not the same as requires_grad = False; it won't automatically be moved to GPU when called with .to()

    def forward(self, x):
        B, T, E = x.size() # batch_size, token_length (always block_size?), emb_dimensions
        # during training, T = block_size always. Could change in inference.
        qkv = self.qkv(x) # B, T, 3*E
        qkv = qkv.reshape(B, T, 3, E) # split into Q, K, V
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.per_head_dim) # split into heads

        
        try:
            q = qkv[:, :, 0, :, :] # Try!
            print(f"qkv[:, :, 0, :, :] is of size {q.size()}")
        except:
            print("q = qkv[:, :, 0, :, :] doesn't work.")

        q, k, v = qkv.unbind(dim = 2) # B, T, self.num_heads, self.per_head_dim
        
        # we need shape (batch_size, num_heads, token_length (T), per_head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        try:
            q_, k_, v_ = qkv.transpose(1,2).unbind(dim = 1)
            assert q_ == q
            assert k_ == k
            assert v_ == v
            print("qkv can be made smaller.")
        except:
            print("didn't work")
            print(f"q_ is shaped {q_.size()} while q is {q.size()}")
        
        k_t = k.transpose(-2, -1) # change to (batch_size, num_heads, .. transpose for mult)
        sdpa_attention = (q @ k_t) * self.scale # T x d * d x T = (batch_size, num_heads, T, T)

        # apply causal_mask
        sdpa_attention = sdpa_attention.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # figure this out better.
        sdpa_attention = F.softmax(sdpa_attention, dim = -1) # along last T: (batch, heads, T, T)
        sdpa_attention = self.dropout(sdpa_attention) # what this mean?
        
        out = sdpa_attention @ v # (batch_size, num_heads, T, per_head_dim)
        out = out.transpose(1, 2) # (batch_size, T, num_heads, per_head_dim)
        # collate all heads again
        out = out.contiguous().reshape(B, T, E)

        out = self.proj(out) # (batch_size, T, E)
        return out



   class MLP(nn.Module):
       def __init__(self, embedding_dim, hidden_dim, dropout = 0.4):
           super().__init__()
           self.net = nn.Sequential(
                   nn.Linear(embedding_dim, hidden_dim),
                   nn.GELU(),
                   nn.Linear(hidden_dim, embedding_dim),
                   nn.Dropout(dropout)
                   )

        def forward(self, x):
            return self.net(x)


    class Transformer(nn.Module):
        def __init__(self, embedding_dim, num_heads, block_size, dropout = 0.4):
            super().__init__()
            self.layer_norm_1 = nn.LayerNorm(embed_dim)
            self.layer_norm_2 = nn.LayerNorm(embed_dim)
            self.attention = CausalSelfAttention(emebdding_dim, vocab_size, num_heads, block_size, dropout)
            self.ffwd = MLP(embedding_dim, 4*embedding_dim, dropout)

        def forward(self,x):
            x = x + self.attention(self.layer_norm_1(x)) # residual connections
            x = x+ self.ffwd(self.layer_norm_2(x)) # also residual connection
            return x

    class GPT(nn.Module):
        def __init__(self, vocab_size, block_size, embedding_dim = 128, num_heads = 4, num_layers = 4, dropout = 0.1):
            super().__init__()
            self.block_size = block_size
            self.embedding_dim = embedding_dim
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
            # check why 1?
            self.dropout = nn.Dropout(dropout)

            self.transformer_blocks = nn.Sequential([Transformer(embedding_dim, num_heads, block_size, dropout) for _ in range(num_layers)])

            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.head = nn.Linear(embed_dim, vocab_size)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean = 0.0, std = 0.01)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)


        def forward(self, idx, targets = None):
            B, T = idx.size()
            assert T <= self.block_size, "Sequence length shouldn't be more than model block_size: Beyond current context capacity."
            token_embeddings = self.token_embeddings(idx)
            position_embeddings = self.pos_embeddings[:, :T, :] # what?
            x = self.drop(token_embeddings + position_embeddings)
            x = self.transformer_blocks(x)
            x = self.layer_norm(x)
            logits = self.head(x)

            if targets is None:
                return logits, None # when in inference

            B, T, E = logits.size()
            logits = logits.view(B*T, E)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        


