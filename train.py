import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer

from models.transformer import Transformer
from data.dataset import BilingualDataset

def train_model():
    # 1. Device configuration (uses GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 2. Load trained tokenizers
    tokenizer_en = Tokenizer.from_file("data/tokenizer_en.json")
    tokenizer_it = Tokenizer.from_file("data/tokenizer_it.json")

    src_vocab_size = tokenizer_en.get_vocab_size()
    tgt_vocab_size = tokenizer_it.get_vocab_size()

    # 3. Load OPUS Books dataset
    print("Loading dataset from Hugging Face...")
    raw_dataset = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']

    # Split into Train (90%) and Validation (10%)
    train_size = int(0.9 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_raw, val_raw = random_split(raw_dataset, [train_size, val_size])

    max_len = 128  # Sequence length constraint for efficient batching

    train_ds = BilingualDataset(train_raw, tokenizer_en, tokenizer_it, "en", "it", max_len)
    val_ds = BilingualDataset(val_raw, tokenizer_en, tokenizer_it, "en", "it", max_len)

    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 4. Initialize the Transformer Model (~75M parameters with these settings)
    model = Transformer(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size,
        d_model=768,
        num_layers=8,
        num_heads=12,
        d_ff=3072,
        max_len=max_len
    ).to(device)

    # 5. Loss and Optimizer setup
    # Label smoothing helps prevent overconfidence and improves generalization
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)

    # 6. Training Loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            encoder_input = batch['encoder_input'].to(device)     # (B, SeqLen)
            decoder_input = batch['decoder_input'].to(device)     # (B, SeqLen)
            encoder_mask = batch['encoder_mask'].to(device)       # (B, 1, 1, SeqLen)
            decoder_mask = batch['decoder_mask'].to(device)       # (B, 1, SeqLen, SeqLen)
            label = batch['label'].to(device)                     # (B, SeqLen)

            # Forward pass through encoder and decoder
            enc_output = model.encode(encoder_input, encoder_mask)
            dec_output = model.decode(decoder_input, enc_output, encoder_mask, decoder_mask)
            proj_output = model.project(dec_output)               # (B, SeqLen, tgt_vocab_size)

            # Compute loss against expected labels
            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            
            # Backpropagation & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_dataloader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")

    # Save trained model weights
    torch.save(model.state_dict(), "transformer_en_it.pt")
    print("Training complete! Model weights saved to 'transformer_en_it.pt'. 🎉")

if __name__ == "__main__":
    train_model()