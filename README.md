

<h1 align="center">
  ⚡ Seq2Seq Transformer
</h1>

<p align="center">
  <strong>English → Italian Neural Machine Translation</strong>
  <br>
  Built from scratch with PyTorch
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
</p>

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/b14d09a2-7eaf-4af0-a722-34eafd9369b1" />


##  The Mission

What happens when you build a Transformer from the ground up?

This project implements the complete **Encoder–Decoder Transformer architecture** in PyTorch, inspired by the landmark paper *Attention Is All You Need*.

Trained on the OPUS Books English–Italian dataset, the model learns to generate Italian translations from English input using self-attention, cross-attention, and autoregressive decoding.

No high-level Transformer model abstraction — the core architecture is implemented from scratch.

##  Under the Hood

The Transformer consists of two major components:

**Encoder — Understanding the Input**
- Token embeddings and sinusoidal positional encodings
- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Padding masks

**Decoder — Generating the Translation**
- Target token embeddings and positional encodings
- Masked multi-head self-attention
- Encoder–decoder cross-attention
- Feed-forward networks
- Residual connections and layer normalization
- Causal masks for autoregressive generation

** Core Mechanisms**
- Scaled dot-product attention
- Multi-head attention
- Causal and padding masks
- Greedy autoregressive decoding

<img width="1241" height="440" alt="Screenshot 2026-10-18 082101" src="https://github.com/user-attachments/assets/a80a6160-ef1f-4891-91d4-52a7388d9d21" />
<img width="1363" height="643" alt="Screenshot 2026-10-18 090434" src="https://github.com/user-attachments/assets/b40bec2e-47e0-4363-a92e-10d9689eadb5" />
<img width="1363" height="645" alt="Screenshot 2026-10-18 091942" src="https://github.com/user-attachments/assets/a7a72dcf-0203-4175-8fc1-dcbb1a539986" />
<img width="1359" height="578" alt="Screenshot 2026-10-18 100310" src="https://github.com/user-attachments/assets/3f5fb796-dccf-475b-9412-f4f4027d2cb6" />
<img width="1366" height="584" alt="Screenshot 2026-10-18 100325" src="https://github.com/user-attachments/assets/dc664834-600b-4c16-9ea3-cf7131670083" />
<img width="1244" height="408" alt="Screenshot 2026-10-18 102406" src="https://github.com/user-attachments/assets/64495c73-03ba-4643-a58e-3b2a1135cbd3" />
<img width="1202" height="629" alt="Screenshot 2026-10-18 104032" src="https://github.com/user-attachments/assets/1fb92f20-1f99-47f0-b575-2cd1f794d7ef" />
<img width="1000" height="514" alt="Screenshot 2026-10-18 110328" src="https://github.com/user-attachments/assets/39217b08-cc03-464d-b2b8-4fd44c089bfb" />
<img width="1208" height="553" alt="Screenshot 2026-10-18 110402" src="https://github.com/user-attachments/assets/b6e94da5-81e7-43b4-85d4-b792e0e52458" />
<img width="1276" height="517" alt="Screenshot 2026-10-18 110420" src="https://github.com/user-attachments/assets/3dc373a0-eb63-46d3-a7a0-188f667c2af5" />


## Architecture at a Glance

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

##  Project Highlights

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

##  Dataset

**OPUS Books — English to Italian**

The model is trained on approximately 32,000 parallel sentence pairs.

Each pair consists of an English sentence and its corresponding Italian translation. Custom word-level tokenizers prepare the text for training and generation.

🔗 Dataset: [Hugging Face — OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)

##  Evaluation & Interpretability

Translation quality is evaluated using three metrics:

- **BLEU** — measures n-gram overlap with reference translations.
- **WER (Word Error Rate)** — measures word-level insertion, deletion, and substitution errors.
- **CER (Character Error Rate)** — measures character-level errors.

Attention maps are visualized to inspect how different attention heads distribute their focus across tokens.

> Actual evaluation scores and sample translations can be added here from the model's validation results.

##  Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **GPU Acceleration:** CUDA
- **Dataset & Tokenization Tools:** Hugging Face Datasets and Tokenizers
- **Experiment Visualization:** TensorBoard

## Getting Started

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

##  What I Learned

- Building the Transformer architecture from first principles.
- Understanding scaled dot-product attention and multi-head attention.
- Implementing encoder–decoder cross-attention.
- Handling causal masking and padding masks.
- Training a neural machine translation model with PyTorch and CUDA.
- Evaluating generated translations and visualizing attention patterns.

##  References

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
