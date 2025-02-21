import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import os


###############################


def download_data(
    url="https://www.gutenberg.org/files/1661/1661-0.txt", filename="input.txt"
):
    """
    By default, use Bible corpus.
    """

    if not os.path.exists(filename):
        print(f"Downloading dataset from {url} to {filename}.")
        r = requests.get(url)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(r.text)

    else:
        print("Dataset already exists. Skipping download.")


download_data()

text = open("input.txt", encoding="utf-8").read()
print(text[4000:4250])
print("... excerpt from training data.")


# character-level tokenizer. TODO: Replace with BPE.

chars = sorted(list(set(text)))  # vocab
vocab_size = len(chars)  # v
print(f"vocab_size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}  # str to int
itos = {i: ch for i, ch in enumerate(chars)}  # int to str


def encode(s):
    return [stoi[c] for c in s]


def decode(s):
    return [itos[i] for i in s]


# encode the entire text:
data = torch.tensor(encode(text), dtype=torch.long)


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
        dummy_x = self.data[idx : idx + block_size]
        dummy_y = self.data[idx + 1 : idx + block_size + 1]
        return dummy_x, dummy_y


# hyperparameters
block_size = 64
batch_size = 2048

dataset = BibleText(data, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model_definition


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_heads, block_size, dropout=0.4):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads  # h
        self.per_head_dim = embedding_dim // num_heads  # d
        self.scale = self.per_head_dim ** (-0.5)  # scaled dpa
        self.embedding_dim = embedding_dim

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)  # expand and split.
        self.proj = nn.Linear(embedding_dim, embedding_dim)  # expression
        self.dropout = nn.Dropout(dropout)

        # masking for causal attention
        # mask the future tokens from the nodes in the graph
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0),
        )
        # register_buffer = part of module's state, but not learnable.
        # Not the same as requires_grad = False; it won't automatically be moved to GPU when called with .to()

    def forward(self, x):
        B, T, E = (
            x.size()
        )  # batch_size, token_length (always block_size?), emb_dimensions
        # during training, T = block_size always. Could change in inference.
        qkv = self.qkv(x)  # B, T, 3*E
        qkv = qkv.reshape(B, T, 3, E)  # split into Q, K, V
        qkv = qkv.reshape(
            B, T, 3, self.num_heads, self.per_head_dim
        )  # split into heads

        # try:
        # q = qkv[:, :, 0, :, :] # Try!
        # print(f"qkv[:, :, 0, :, :] is of size {q.size()}")
        # except:
        #   pass
        # print("q = qkv[:, :, 0, :, :] doesn't work.")

        q, k, v = qkv.unbind(dim=2)  # B, T, self.num_heads, self.per_head_dim

        # we need shape (batch_size, num_heads, token_length (T), per_head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # try:
        #    q_, k_, v_ = qkv.transpose(1,2).unbind(dim = 1)
        #    assert q_ == q
        #    assert k_ == k
        #    assert v_ == v
        #   print("qkv can be made smaller.")
        # except:
        #    pass
        # print("didn't work")
        # print(f"q_ is shaped {q_.size()} while q is {q.size()}")

        k_t = k.transpose(
            -2, -1
        )  # change to (batch_size, num_heads, .. transpose for mult)
        sdpa_attention = (
            q @ k_t
        ) * self.scale  # T x d * d x T = (batch_size, num_heads, T, T)

        # apply causal_mask
        sdpa_attention = sdpa_attention.masked_fill(
            self.mask[:, :, :T, :T] == 0, float("-inf")
        )
        # figure this out better.
        sdpa_attention = F.softmax(
            sdpa_attention, dim=-1
        )  # along last T: (batch, heads, T, T)
        sdpa_attention = self.dropout(sdpa_attention)  # what this mean?

        out = sdpa_attention @ v  # (batch_size, num_heads, T, per_head_dim)
        out = out.transpose(1, 2)  # (batch_size, T, num_heads, per_head_dim)
        # collate all heads again
        out = out.contiguous().reshape(B, T, E)

        out = self.proj(out)  # (batch_size, T, E)
        return out


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_experts=4, dropout=0.4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [MLP(embedding_dim, hidden_dim, dropout) for _ in range(self.num_experts)]
        )
        self.gate = nn.Linear(embedding_dim, num_experts)

    def forward(self, x):
        B, T, E = x.shape
        gate_logits = self.gate(x)  # batch_size, token_length, num_experts
        gate_probs = F.softmax(
            gate_logits, dim=-1
        )  # batch_size, token_length, num_experts

        top_indices = torch.argmax(gate_probs, dim=-1)  # batch_size, token_length

        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            mask = top_indices == expert_id  # batch_size, token_length (Boolean)

            x_expert = x[mask]
            # mask all irrelevant tokens for this expert
            expert_out = self.experts[expert_id](
                x_expert
            )  # pass the modified, masked token train to the expert
            # no scaling added
            output[mask] = (
                expert_out  # replace the parts of the output that this expert handles
            )
        return output


class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout=0.4):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim, vocab_size, num_heads, block_size, dropout
        )
        # self.ffwd = MLP(embedding_dim, 4*embedding_dim, dropout)
        self.ffwd = MoEFeedForward(embedding_dim, num_heads, block_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))  # residual connections
        x = x + self.ffwd(self.layer_norm_2(x))  # also residual connection
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.num_heads = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.per_head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.per_head_dim ** (-0.5)

    def forward(self, x):
        B, T, E = x.shape
        qkv = self.qkv(x).reshape(
            B, T, 3, self.d_model
        )  # batch_size, token_length, 3, d_model
        qkv = qkv.reshape(
            B, T, 3, num_heads, per_head_dim
        )  # batch_size, token_length, 3, num_heads, per_head_dim

        q, k, v = qkv.unbind(dim=2)  # batch_size, token_length, num_heads, per_head_dim
        q = q.transpose(
            1, 2
        )  # batch_size, num_heads, token_length, per_head_dim [arrange along num_heads]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.transpose(
            -1, -2
        )  # transpose to multiply with q at the last two dims. we want token x token
        attn_weights = F.softmax(
            (q @ k) * self.scale, dim=-1
        )  # batch, num_head, token_length, token_length
        attn_weights = self.dropout(attn_weights)

        dpa_matrix = attn_weights @ v  # batch, num_head, token_length, per_head_dim
        dpa_matrix = dpa_matrix.transpose(1, 2).reshape(B, T, E)
        return self.proj(dpa_matrix)


class TokenTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffwd, dropout=0.1):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_ffwd),
            nn.GELU(),
            nn.Linear(d_ffwd, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        fgram_max_n=3,
        fgram_nhead=2,
        fgram_ffwd_dim=64,
        fgram_set={"av", "ion", "on", "is", "was", "at", "ark", "be", "tri"},
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
        # check why 1?
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                Transformer(embedding_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

        self.fgram_transformer = TokenTransformer(
            embedding_dim, fgram_nhead, fgram_ffwd_dim, dropout
        )
        self.fgram_max_n = fgram_max_n
        self.fgram_set = fgram_set

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.block_size
        ), "Sequence length shouldn't be more than model block_size: Beyond current context capacity."
        token_embeddings = self.token_embedding(
            idx
        )  # (bathc_size, token_length, emb_size)
        token_embeddings = self.compute_f_gram_embeddings(idx, token_embeddings)
        position_embeddings = self.pos_embedding[
            :, :T, :
        ]  # what? (batch_size, token_length, emb_size)
        x = self.dropout(token_embeddings + position_embeddings)
        x = self.transformer_blocks(x)
        # print(f"x is {x.size()}")
        # print(f"x is {B, T, self.embedding_dim}")
        x = self.layer_norm(x)
        logits = self.head(x)  # (batch_size, token_length, vocab_size)

        if targets is None:
            return logits, None  # when in inference

        B, T, V = logits.size()
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            # crop if longer than block-size
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self.forward(idx_cond)
            # See last token
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def compute_f_gram_embeddings(self, idx, base_embeddings):
        B, T, E = base_embeddings.size()
        final_embeddings = base_embeddings.clone()

        for n_batch in range(B):
            tokens = idx[n_batch].tolist()
            for i in range(T):
                for n in range(self.fgram_max_n, 1, -1):
                    if i - n + 1 < 0:
                        continue  # last f-gram size
                    candidate_token = tuple(tokens[i - n + 1 : i + 1])
                    if candidate_token in self.fgram_set:
                        span_embeds = base_embeddings[
                            n_batch, i - n + 1 : i + 1, :
                        ]  # pull all embeddings of size n
                        transformed = self.fgram_transformer(
                            span_embeds.unsqueeze(0)
                        )  # add batch_dim
                        contextualized_emb = transformed[
                            0, -1, :
                        ]  # take the last token, or just output to one embd
                        final_embeddings[b, i, :] = contextualized_emb
                        break
        return final_embeddings


def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):
        losses = []
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} loss = {loss.item():.4f}")
        print(f"Epoch {epoch+1} loss: {sum(losses)/len(losses):.4f}")


def main():
    device = torch.device("cuda")

    embedding_dim = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    fgram_set = {
        "th",
        "he",
        "in",
        "er",
        "an",
        "re",
        "nd",
        "at",
        "on",
        "nt",
        "ha",
        "es",
        "st",
        "en",
        "ed",
        "to",
        "it",
        "ou",
        "ea",
        "hi",
        "is",
        "or",
        "ti",
        "as",
        "te",
        "et",
        "ng",
        "of",
        "al",
        "de",
        "se",
        "le",
        "sa",
        "si",
        "ar",
        "ve",
        "ra",
        "ld",
        "ur",
        "li",
        "ri",
        "ic",
        "ne",
        "co",
        "ce",
        "ma",
        "el",
        "ta",
        "la",
        "ro",
        "io",
        "be",
        "wi",
        "ho",
        "na",
        "me",
        "di",
        "om",
        "no",
        "we",
        "us",
        "mi",
        "il",
        "ch",
        "mo",
        "im",
        "ca",
        "ot",
        "il",
        "fo",
        "pe",
        "lo",
        "ad",
        "un",
        "so",
        "po",
        "ho",
        "fa",
        "ir",
        "pa",
        "si",
        "su",
        "do",
        "vi",
        "pr",
        "wh",
        "fr",
        "pl",
        "tr",
        "cl",
        "sh",
        "bl",
        "fl",
        "gr",
        "sp",
        "sc",
        "br",
        "gl",
        "cr",
        "sl",
        "tw",
        "dr",
        "sw",
        "qu",
        "sm",
        "sk",
        "ph",
        "wr",
        "kn",
        "gn",
        "dw",
        "hy",
        "cy",
        "fy",
        "gy",
        "ly",
        "my",
        "ny",
        "py",
        "ry",
        "sy",
        "ty",
        "vy",
        "zy",
        "by",
        "dy",
        "ky",
        "ty",
        "vy",
        "zy",
        "ay",
        "ey",
        "oy",
        "uy",
        "io",
        "ia",
        "ie",
        "ei",
        "ai",
        "au",
        "ou",
        "eu",
        "ui",
        "ue",
        "oa",
        "oe",
        "oo",
        "ua",
        "ue",
        "uo",
        "aa",
        "ee",
        "ii",
        "oo",
        "uu",
        "bb",
        "cc",
        "dd",
        "ff",
        "gg",
        "hh",
        "jj",
        "kk",
        "ll",
        "mm",
        "nn",
        "pp",
        "qq",
        "rr",
        "ss",
        "tt",
        "vv",
        "ww",
        "xx",
        "yy",
        "zz",
        "the",
        "and",
        "tha",
        "ent",
        "ing",
        "ion",
        "tio",
        "for",
        "nde",
        "has",
        "nce",
        "edt",
        "tis",
        "oft",
        "sth",
        "men",
        "her",
        "ere",
        "his",
        "ver",
        "all",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
        "ect",
        "com",
        "per",
        "sta",
        "ear",
        "art",
        "ess",
        "not",
        "ive",
        "rea",
        "com",
        "eve",
        "ove",
        "our",
        "but",
        "are",
        "one",
        "oul",
        "ver",
        "out",
        "hin",
        "wit",
        "thi",
        "ter",
        "est",
        "ers",
        "ati",
        "hat",
        "ate",
        "rea",
        "con",
        "int",
        "res",
        "pro",
    }

    model = GPT(
        vocab_size,
        block_size,
        embedding_dim,
        num_heads,
        num_layers,
        dropout,
        fgram_set=fgram_set,
    )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    train(model, dataloader, optimizer, device, epochs=1)
    model.eval()
    context = "Watson was sitting by the armchair, and \n"

    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)
    generated = model.generate(
        context_encoded, max_new_tokens=500, temperature=0.8, top_k=10
    )
    [print(x, end="") for x in (decode(generated[0].tolist()))]

    train(model, dataloader, optimizer, device, epochs=3)
    context = "Watson was sitting by the armchair, and..."

    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)
    generated = model.generate(context_encoded, max_new_tokens=500)
    [print(x, end="") for x in (decode(generated[0].tolist()))]

    train(model, dataloader, optimizer, device, epochs=3)
    context = "Watson was sitting by the armchair, and..."

    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)
    generated = model.generate(context_encoded, max_new_tokens=200)
    [print(x, end="") for x in (decode(generated[0].tolist()))]


if __name__ == "__main__":
    main()
