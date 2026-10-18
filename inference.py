import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from models.transformer import Transformer
from data.dataset import causal_mask

def greedy_decode(model, src, src_mask, max_len, bos_idx, eos_idx, device):
    """
    Generates target tokens one by one autoregressively.
    """
    model.eval()
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        # Initialize decoder input with BOS token
        decoder_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            dec_output = model.decode(decoder_input, enc_output, src_mask, tgt_mask)
            prob = model.project(dec_output[:, -1])
            
            _, next_word = torch.max(prob, dim=1)
            
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_word.item()]], dtype=torch.long, device=device)], dim=1
            )
            
            if next_word.item() == eos_idx:
                break
                
    return decoder_input.squeeze(0)

def visualize_attention(encoder_tokens, decoder_tokens, attention_weights, layer_idx=0, head_idx=0):
    """
    Plots a heatmap of the cross-attention weights between English and Italian words.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract weights for specific layer and head: Shape (SeqLen_tgt, SeqLen_src)
    attn = attention_weights[layer_idx][head_idx].detach().cpu().numpy()
    
    sns.heatmap(
        attn[:len(decoder_tokens), :len(encoder_tokens)],
        xticklabels=encoder_tokens,
        yticklabels=decoder_tokens,
        cmap="viridis",
        annot=True,
        ax=ax
    )
    
    plt.xlabel("Encoder Input (English)")
    plt.ylabel("Decoder Output (Italian)")
    plt.title(f"Cross-Attention Map (Layer {layer_idx+1}, Head {head_idx+1})")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("attention_map.png")
    print("Attention heatmap saved as 'attention_map.png'! 📊")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load tokenizers
    tokenizer_en = Tokenizer.from_file("data/tokenizer_en.json")
    tokenizer_it = Tokenizer.from_file("data/tokenizer_it.json")
    
    # 2. Load trained model weights
    src_vocab_size = tokenizer_en.get_vocab_size()
    tgt_vocab_size = tokenizer_it.get_vocab_size()
    
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        d_model=768, 
        num_layers=8, 
        num_heads=12, 
        d_ff=3072, 
        max_len=128  # <-- Add this parameter here!
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("transformer_en_it.pt", map_location=device))
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print("Model weights file 'transformer_en_it.pt' not found. Run train.py first!")
        exit()

    # 3. Test Translation on a Sample Sentence
    sample_sentence = "The cat sat on the mat."
    print(f"\nTranslating English: '{sample_sentence}'")
    
    # Tokenize input
    tokens = tokenizer_en.encode(sample_sentence).ids
    src = torch.cat([
        torch.tensor([2], dtype=torch.long),  # BOS
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor([3], dtype=torch.long)   # EOS
    ]).unsqueeze(0).to(device)
    
    src_mask = (src != 0).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Run greedy decode
    translated_tokens = greedy_decode(model, src, src_mask, max_len=50, bos_idx=2, eos_idx=3, device=device)
    translated_text = tokenizer_it.decode(translated_tokens.tolist())
    
    print(f"Translated Italian: {translated_text}")