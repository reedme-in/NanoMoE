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
        

    def forward(self, x):
        B, T, E = x.size() # batch_size, token_length (always block_size?), emb_dimensions
        # during training, T = block_size always. Could change in inference.

