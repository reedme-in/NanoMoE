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



