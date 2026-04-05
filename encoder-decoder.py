### Importing necessary packages
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn
from datasets import load_dataset
import sentencepiece as spm
from torch.utils.data import DataLoader
import os

@dataclass
class TranslationConfig:
    context_length = 32 ### Specifying the Context Length which the Transformer will be available to Process
    vocab_size = 16000 ### Vocab Size -- Tiktoken Vocab Size -- 50,257 also include <|endoftext|>, print(tiktoken.get_encoding('gpt2').n_vocab)
    n_layer = 4 ### No of the Transformer Decoder stacks
    n_head = 4 ### No of Attention Heads Used
    n_embd = 128 ### Dimensionality of the Word Embedding

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True, dropout=0.2)
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
        self.layer_norm2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, attn_mask=None)
        x = x + attn_out
        x = self.layer_norm1(x)
        mlp_out = self.multi_layer_perceptron(x)
        x = x + mlp_out
        x = self.layer_norm2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True, dropout=0.2)
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True, dropout=0.2)
        self.layer_norm3 = nn.LayerNorm(config.n_embd)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), ### Activation Used as Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
    def forward(self, x, encoder_output):
        B, T, C = x.shape ## Getting the Batch, time and Channel dimension from the input
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool() ## Adding the Mask -- Disabling the Upper Triangle.
        attn_out, _ = self.attn(x, x, x, attn_mask=mask) ## It returns Attention Weights too
        x = x + attn_out ## Applied Residual Masked Multihead Attention with Causality
        x = self.layer_norm1(x)
        attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=None)
        x = x + attn_out
        x = self.layer_norm2(x)
        mlp_out = self.multi_layer_perceptron(x)
        x = x + mlp_out
        x = self.layer_norm3(x)
        return x
    
class Translation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            token_embedding_decoder = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embedding_decoder = nn.Embedding(config.context_length, config.n_embd),
            decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            layer_norm = nn.LayerNorm(config.n_embd),
            token_embedding_encoder = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embedding_encoder = nn.Embedding(config.context_length, config.n_embd),
            encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, src, tgt, targets=None):
        B, T_src = src.shape
        src_tok_emb = self.transformer.token_embedding_encoder(src)
        src_pos = torch.arange(0, T_src)
        src_pos_emb = self.transformer.pos_embedding_encoder(src_pos)

        enc_x = src_tok_emb + src_pos_emb
        for Block in self.transformer.encoder:
            enc_x = Block(enc_x) ### Passing through all 4 Encoder Block
        encoder_output = enc_x

        B, T_tgt = tgt.shape
        tgt_tok_emb = self.transformer.token_embedding_decoder(tgt)
        tgt_pos = torch.arange(0, T_tgt)
        tgt_pos_emb = self.transformer.pos_embedding_decoder(tgt_pos)

        dec_x = tgt_tok_emb + tgt_pos_emb
        for Block in self.transformer.decoder:
            dec_x = Block(dec_x, encoder_output)

        dec_x = self.transformer.layer_norm(dec_x)
        logits = self.lm_head(dec_x)
        loss = None

        if targets is not None:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=PAD)

        return logits, loss


dataset = load_dataset("cfilt/iitb-english-hindi")
if not os.path.exists("spm.model"):
    with open("corpus.txt", "w", encoding="utf-8") as f:
        for ex in dataset["train"]:
            f.write(ex["translation"]["en"] + "\n")
            f.write(ex["translation"]["hi"] + "\n")

    spm.SentencePieceTrainer.train(
        input="corpus.txt",
        model_prefix="spm",
        vocab_size=16000
    )

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

BOS = sp.bos_id()
EOS = sp.eos_id()
PAD = 0   # safe choice
MAX_LEN = 32

# Preprocessing
def encode(example):
    en = example["translation"]["en"]
    hi = example["translation"]["hi"]

    src = sp.encode(en)[:MAX_LEN-1] + [EOS]

    tgt = [BOS] + sp.encode(hi)[:MAX_LEN-2] + [EOS]
    tgt_input = tgt[:-1]
    targets = tgt[1:]

    src += [PAD] * (MAX_LEN - len(src))
    tgt_input += [PAD] * (MAX_LEN - len(tgt_input))
    targets += [PAD] * (MAX_LEN - len(targets))

    return {
        "src": torch.tensor(src),
        "tgt": torch.tensor(tgt_input),
        "targets": torch.tensor(targets)
    }

tokenized_dataset = dataset["train"].map(encode)
tokenized_dataset.set_format(type="torch", columns=["src", "tgt", "targets"])

loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# Training

model = Translation(TranslationConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(3):
    for batch in loader:
        src = batch["src"]
        tgt = batch["tgt"]
        targets = batch["targets"]

        _, loss = model(src, tgt, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")