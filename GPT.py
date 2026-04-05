### Importing necessary packages
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn
import tiktoken
import requests

@dataclass
class GPTConfig:
    context_length = 32 ### Specifying the Context Length which the Transformer will be available to Process
    vocab_size = 50257 ### Vocab Size -- Tiktoken Vocab Size -- 50,257 also include <|endoftext|>, print(tiktoken.get_encoding('gpt2').n_vocab)
    n_layer = 4 ### No of the Transformer Decoder stacks
    n_head = 4 ### No of Attention Heads Used
    n_embd = 128 ### Dimensionality of the Word Embedding 


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True, dropout=0.2)
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
    def forward(self, x):
        B, T, C = x.shape ## Getting the Batch, time and Channel dimension from the input
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool() ## Adding the Mask -- Disabling the Upper Triangle.
        attn_out, _ = self.attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x), attn_mask=mask) ## It returns Attention Weights too
        x = x + attn_out ## Applied Residual Masked Multihead Attention with Causality

        x = x + self.multi_layer_perceptron(self.layer_norm2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embedding = nn.Embedding(config.context_length, config.n_embd),
            decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            layer_norm = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer.token_embedding(idx)
        pos = torch.arange(0, T)
        pos_emb = self.transformer.pos_embedding(pos)

        x = tok_emb + pos_emb
        for Block in self.transformer.decoder:
            x = Block(x) ### Passing through all 4 Decoder Blocks
        
        x = self.transformer.layer_norm(x)
        logits = self.lm_head(x) ### B, T, n_embd ----> B, T, Vocab_Size

        loss = None

        if targets is not None:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
        
### Fetching the DataSet
train_url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt'
val_url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt'
train_text = requests.get(url=train_url).text
val_text = requests.get(url=val_url).text

### Encoding
enc = tiktoken.get_encoding("gpt2")
train_tokens = enc.encode(train_text)
val_tokens = enc.encode(val_text)
train_data = torch.tensor(train_tokens, dtype=torch.long)
val_data = torch.tensor(val_tokens, dtype=torch.long)

### Function to get Batches of Data to Train the Model
def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    
    return x, y

### Training the Model
model = GPT(GPTConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


for step in range(4000):
    x, y = get_batch(train_data, batch_size=16, context_length=model.config.context_length)

    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if step % 200 == 0:
        print(f"Step {step}: Loss {loss.item()}")


@torch.no_grad()
def evaluate(model, data):
    x, y = get_batch(data=data, batch_size=16, context_length=model.config.context_length)
    _, loss = model(x, y)
    return loss.item()

print(f"Validation Loss: {evaluate(model=model, data=val_data)}")

### Sampling from the Model
def generate(model, idx, max_new_token):
    for _ in range(max_new_token):
        idx_cond = idx[:, -model.config.context_length:]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, next_token), dim=1)
    
    return idx

start = torch.tensor([enc.encode("The meaning of life is")])
output = generate(model, start, 50)
print(f"\033[92m{enc.decode(output[0].tolist())}\033[0m")