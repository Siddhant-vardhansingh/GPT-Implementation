# 🧠 GPT Implementation (PyTorch)

A minimal, end-to-end implementation of a **GPT-style language model** built using PyTorch.
This project covers everything from **data loading → tokenization → model architecture → training → text generation**.

---

## Features:

- GPT.py: Implements a decoder-only Transformer with 4 layers and 4 attention heads. Uses `torch.nn.MultiheadAttention` for masked self-attention.
- GPT-v2.py: Implements the same architecture but does not use `torch.nn.MultiheadAttention`. Instead, it implements masked self-attention from scratch using `torch.nn.Linear` layers and manual masking.
- encoder-decoder.py: Implements a full Transformer architecture with both encoder and decoder blocks. The encoder processes the input sequence, while the decoder generates the output sequence using masked self-attention and cross-attention to the encoder outputs.

---

## 🚀 Overview

This project implements a **decoder-only Transformer (GPT)** using PyTorch APIs:

- Masked Multi-Head Self Attention (Causal Masking)
- Positional + Token Embeddings
- Residual Connections + LayerNorm
- Feedforward (MLP) layers
- Cross Entropy Loss for language modeling
- Training on **WikiText-2 dataset**
- Text generation using sampling

---

## 🏗️ Architecture

### 🔹 Model Components

- **Embedding Layer**
  - Token Embedding
  - Positional Embedding

- **Decoder Blocks (Stacked 4 times)**
  - LayerNorm
  - Masked Multi-Head Attention (Causal Masking, using `torch.nn.MultiheadAttention` and 4 attention heads)
  - Feedforward Neural Network (MLP)
  - Residual Connections

- **Final LayerNorm + Linear Head**
  - Projects to vocabulary size

---

## 📊 Dataset

We use **WikiText-2**, a cleaned Wikipedia dataset.

- Source: PyTorch examples repo
- Format: Raw text
- Tokenization: GPT-2 tokenizer via `tiktoken`

---

## 🔧 Setup

### 1. Install dependencies

```bash
pip install torch tiktoken requests
```

---

### 2. Run the script

```bash
python GPT.py
```

---

## 🧪 Training Pipeline

### 1. Data Loading

```python
train_text = requests.get(train_url).text
```

---

### 2. Tokenization

```python
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(train_text)
```

---

### 3. Batch Sampling

Random chunks of fixed context length are sampled:

```python
x = data[i : i + context_length]
y = data[i+1 : i + context_length + 1]
```

---

### 4. Training Loop

- Optimizer: AdamW
- Loss: Cross Entropy
- Random batch sampling each step

---

## 📈 Loss Tracking

- Training loss is logged every few steps
- Loss is averaged over multiple batches to reduce noise
- Visualization using `matplotlib`

---

## 📝 Text Generation

After training, the model can generate text:

```python
start = torch.tensor([enc.encode("The meaning of life is")])
output = generate(model, start, 50)
print(enc.decode(output[0].tolist()))
```

### ⚙️ How it works:

- Uses **autoregressive generation**
- Predicts next token step-by-step
- Uses **softmax + sampling**

---

## 🧠 Key Concepts Implemented

- Causal masking (no future token access)
- Sliding context window
- Autoregressive language modeling
- Token shifting for targets
- Batch randomization for generalization

---

## 📚 References

This implementation is inspired by foundational papers in the field of Transformers and language models:

- **Attention Is All You Need** (Vaswani et al., 2017)
  Introduced the Transformer architecture and the concept of self-attention.
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
  Demonstrated the power of large-scale generative pre-trained transformers.

---

## ⚡ Limitations

- Small model (trained on CPU)
- Limited context length (32 tokens)
- No advanced optimizations (e.g., KV cache, Flash Attention)
- Basic sampling (no temperature/top-k)

---

## 🚀 Future Improvements

- Increase context length
- Add temperature / top-k sampling
- Implement learning rate scheduling
- Add validation tracking during training
- Use GPU for faster training
- Replace attention with `scaled_dot_product_attention`
- Add checkpoint saving/loading

---

## 📌 Example Output

```
The meaning of life is not something we can easily define, but it is often related to...
```

_(Output quality improves with more training)_

---

## 🎯 Learning Outcomes

This project helps you understand:

- How GPT models work internally
- How attention masking is implemented
- How language models are trained
- End-to-end ML system design

---

## 🙌 Acknowledgements

- PyTorch documentation
- OpenAI GPT tokenizer (`tiktoken`)
- WikiText-2 dataset

![alt text](image.png)
