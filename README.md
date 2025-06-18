<div align=center>
<a href = "reedme.in"><img width="500px" height="500px" src= "https://github.com/user-attachments/assets/a1497ccd-eb7e-47f5-b156-a62c4239ee61"></a>
</div>

<div align = center>
  
  ```NanoMoE``` is a minimal rewrite of Karpathy's NanoGPT, from scratch, with pedogical implementations for some new features, such as MOEs.

</div>

-----------------------------------------
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)![Compatibility](https://img.shields.io/badge/compatible%20with-python3.6.x-blue.svg)


A compact, from-scratch character-level Transformer model with Rotary Position Embeddings, Mixture-of-Experts feed-forward layers, and F-gram contextual augmentation — all implemented in a single `model.py` file. Beats out NanoGPT's generative capabilities on a roughly similar LOC (to offset higher memory but retain gains from MOE and RoPE, consider turning of the F-gram context). Run the (badly named) `model.py` file to start training; you might want to change some of the hyperparameters to fit on a consumer GPU. 

Written for practice, manually, on a keyboard in two afternoons.

Currently, it loads the TinyShakespeare dataset; perhaps a switch to FineWeb is warranted.

---

## Configuration (to be added!)

I'll be adding an arg parser soon, but until then, these are rough, recommended values! (Although I assume TinyShakespeare would work no matter what you choose to run on).

| Argument          | Description                                   | Default |
| ----------------- | --------------------------------------------- | ------- |
| `--embedding_dim` | Token embedding size                          | 128     |
| `--num_heads`     | Number of attention heads                     | 4       |
| `--num_layers`    | Number of Transformer blocks                  | 4       |
| `--block_size`    | Context window (sequence length)              | 64      |
| `--dropout`       | Dropout probability                           | 0.1     |
| `--moe_experts`   | Number of experts in the MoE layer            | 4       |
| `--fgram_max_n`   | Maximum n-gram length for F-gram augmentation | 3       |
| `--learning_rate` | AdamW learning rate                           | 3e-4    |
| `--batch_size`    | Batch size                                    | 512     |
| `--epochs`        | Number of training epochs                     | 10      |

---

## Model Architecture

1. **Token Embeddings** + **Positional Embeddings**
2. **F-gram Encoder**

   * Scans up to `fgram_max_n` character n-grams and applies a small Transformer to contextualize matching spans.
3. **Transformer Blocks** × *N*

   * **Rotary Self-Attention**
   * **Mixture-of-Experts Feed-Forward**
4. **Final LayerNorm** → **Linear Output Head**

---

## Extending NanoMoE

* Swap in a subword tokenizer (BPE/WordPiece) in place of the character encoder
* Automate F-gram mining from your corpus rather than hard-coding
* Add validation, checkpointing, and learning-rate schedules
* Integrate mixed-precision (`torch.cuda.amp`) for speed and memory savings

---
