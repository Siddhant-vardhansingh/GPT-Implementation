### Importing necessary packages
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

@dataclass
class BERTConfig:
    context_length = 128 ### Specifying the Context Length which the Transformer will be available to Process
    vocab_size = 50257 ### Vocab Size -- Tiktoken Vocab Size -- 50,257 also include <|endoftext|>, print(tiktoken.get_encoding('gpt2').n_vocab)
    n_layer = 4 ### No of the Transformer Encoder stacks
    n_head = 4 ### No of Attention Heads Used
    n_embd = 128 ### Dimensionality of the Word Embedding 


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True, dropout=0.2)
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        B, T, C = x.shape ## Getting the Batch, time and Channel dimension from the input
        attn_out, _ = self.attn(x, x, x, attn_mask=None) ## It returns Attention Weights too
        x = x + attn_out ## Applied Residual Masked Multihead Attention with Causality
        x = self.layer_norm1(x)
        x = x + self.multi_layer_perceptron(x)
        x = self.layer_norm2(x)
        return x
    
class BERTClassifier(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embedding = nn.Embedding(config.context_length, config.n_embd),
            encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            layer_norm = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, num_labels)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer.token_embedding(idx)
        pos = torch.arange(0, T)
        pos_emb = self.transformer.pos_embedding(pos)

        x = tok_emb + pos_emb
        for Block in self.transformer.encoder:
            x = Block(x) 
        
        x = self.transformer.layer_norm(x)
        class_token = x[:, 0, :] ### The |CLS| Token
        logits = self.lm_head(class_token) 

        loss = None

        if targets is not None:
            B, V = logits.shape
            logits = logits.view(B, V)
            targets = targets.view(B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = BERTClassifier(BERTConfig(), num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

raw_datasets = load_dataset("imdb")
enc = tiktoken.get_encoding("gpt2")

def tokenize_function(examples):
    cls_id = enc.eot_token
    ctx_len = 128 
    
    result = []
    for text in examples["text"]:
        tokens = enc.encode(text)[:ctx_len - 1]
        input_ids = [cls_id] + tokens
        input_ids += [cls_id] * (ctx_len - len(input_ids))
        result.append(input_ids)
    
    return {"input_ids": result, "label": examples["label"]}

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "label"])

train_loader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)

model.train()
for epoch in range(3):
    for batch in train_loader:
        x = batch["input_ids"]
        y = batch["label"]
        
        logits, loss = model(x, targets=y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}", end="\r")

def predict_sentiment(text, model, config, enc):
    model.eval() 
    cls_id = enc.eot_token
    tokens = enc.encode(text)[:config.context_length - 1]
    
    input_ids = [cls_id] + tokens
    padding_len = config.context_length - len(input_ids)
    input_ids += [cls_id] * padding_len
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence = probs[0][prediction].item()
    
    return label, confidence

# --- Example Usage ---
review = "It is the greatest masterpiece, Good Direction and Acting"
label, conf = predict_sentiment(review, model, BERTConfig(), enc)
print(f"Sentiment of the {review}: {label} ({conf*100:.2f}%)")