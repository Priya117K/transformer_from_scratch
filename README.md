# ⚡ Seq2Seq Transformer | English → Italian

<p align="center">
  <strong>From Attention to Understanding.</strong>
  <br>
  A Transformer built from scratch. A journey into neural machine translation.
  <br><br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorBoard">
</p>

---

## 🌌 The Mission

What happens when you build a Transformer from the ground up?

This project implements the complete **Encoder–Decoder Transformer architecture** in PyTorch, inspired by the landmark paper *Attention Is All You Need*.

Trained on the OPUS Books English–Italian dataset, the model learns to generate Italian translations from English input using self-attention, cross-attention, and autoregressive decoding.

No high-level Transformer model abstraction — the core architecture is implemented from scratch.

## 🧠 Under the Hood

The Transformer consists of two major components:

**🔹 Encoder — Understanding the Input**
- Token embeddings and sinusoidal positional encodings
- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Padding masks

**🔹 Decoder — Generating the Translation**
- Target token embeddings and positional encodings
- Masked multi-head self-attention
- Encoder–decoder cross-attention
- Feed-forward networks
- Residual connections and layer normalization
- Causal masks for autoregressive generation

**⚡ Core Mechanisms**
- Scaled dot-product attention
- Multi-head attention
- Causal and padding masks
- Greedy autoregressive decoding

## 🏗️ Architecture at a Glance

```text
             ENGLISH INPUT
                   │
             Tokenization
                   │
            Token Embeddings
                   │
       Sinusoidal Positional Encoding
                   │
                   ▼
          ┌──────────────────┐
          │     ENCODER      │
          │                  │
          │ Multi-Head       │
          │ Self-Attention   │
          │        ↓         │
          │ Add & Norm       │
          │        ↓         │
          │ Feed-Forward     │
          │        ↓         │
          │ Add & Norm       │
          └────────┬─────────┘
                   │
             Encoder Output
                   │
                   ▼
          ┌──────────────────┐
          │     DECODER      │◄──── Previous Target Tokens
          │                  │
          │ Masked Self-     │
          │ Attention        │
          │        ↓         │
          │ Cross-Attention  │◄──── Encoder Output
          │        ↓         │
          │ Feed-Forward     │
          │                  │
          └────────┬─────────┘
                   │
             Linear + Softmax
                   │
                   ▼
            NEXT ITALIAN TOKEN
                   │
             Greedy Decoding
                   │
                   ▼
            ITALIAN OUTPUT
```

## 🚀 Project Highlights

| Component | Implementation |
|---|---|
| Architecture | Encoder–Decoder Transformer |
| Model size | Approximately 75M parameters |
| Dataset | OPUS Books, English → Italian |
| Training data | Approximately 32K sentence pairs |
| Tokenization | Custom word-level tokenizers |
| Optimizer | Adam |
| Loss function | Label-smoothed cross-entropy |
| Positional encoding | Sinusoidal |
| Inference | Greedy autoregressive decoding |
| Hardware | NVIDIA Tesla V100 GPU |
| Evaluation | BLEU, WER, CER |
| Interpretability | Multi-head attention visualizations |

## 📚 Dataset

**OPUS Books — English to Italian**

The model is trained on approximately 32,000 parallel sentence pairs.

Each pair consists of an English sentence and its corresponding Italian translation. Custom word-level tokenizers prepare the text for training and generation.

🔗 Dataset: [Hugging Face — OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)

## 📊 Evaluation & Interpretability

Translation quality is evaluated using three metrics:

- **BLEU** — measures n-gram overlap with reference translations.
- **WER (Word Error Rate)** — measures word-level insertion, deletion, and substitution errors.
- **CER (Character Error Rate)** — measures character-level errors.

Attention maps are visualized to inspect how different attention heads distribute their focus across tokens.

> Actual evaluation scores and sample translations can be added here from the model's validation results.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **GPU Acceleration:** CUDA
- **Dataset & Tokenization Tools:** Hugging Face Datasets and Tokenizers
- **Experiment Visualization:** TensorBoard

## ⚙️ Getting Started

Clone the repository:

```bash
git clone https://github.com/Priya117K/<repository-name>.git
cd <repository-name>
```

Install the dependencies listed in your `requirements.txt`:

```bash
pip install -r requirements.txt
```

Run the training or inference scripts using the commands defined by your implementation.

*Note: Replace the repository name and commands with the actual values from your project.*

## 💡 What I Learned

- Building the Transformer architecture from first principles.
- Understanding scaled dot-product attention and multi-head attention.
- Implementing encoder–decoder cross-attention.
- Handling causal masking and padding masks.
- Training a neural machine translation model with PyTorch and CUDA.
- Evaluating generated translations and visualizing attention patterns.

## 📖 References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [OPUS Books Dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Documentation](https://huggingface.co/docs)

---

<p align="center">
  <strong>Built with curiosity. Powered by attention. ⚡</strong>
  <br><br>
  <a href="https://github.com/Priya117K">Priyadharshini K · GitHub</a>
</p>
