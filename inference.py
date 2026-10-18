from io import BytesIO

import gradio as gr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tokenizers import Tokenizer
from models.transformer import Transformer
from data.dataset import causal_mask

model: Transformer | None = None
device: torch.device | None = None
tokenizer_en: Tokenizer | None = None
tokenizer_it: Tokenizer | None = None


def greedy_decode(model, src, src_mask, max_len, bos_idx, eos_idx, device, repetition_penalty=1.2):
    """
    Generates target tokens one by one autoregressively.
    """
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero.")

    model.eval()
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        # Initialize decoder input with BOS token
        decoder_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            dec_output = model.decode(decoder_input, enc_output, src_mask, tgt_mask)
            logits = model.project(dec_output[:, -1])
            generated_tokens = decoder_input.unique()
            repeated_logits = logits[:, generated_tokens]
            logits[:, generated_tokens] = torch.where(
                repeated_logits < 0,
                repeated_logits * repetition_penalty,
                repeated_logits / repetition_penalty,
            )
            
            _, next_word = torch.max(logits, dim=1)
            
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
    fig.savefig("attention_map.png")
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="png")
    image_buffer.seek(0)
    image = Image.open(image_buffer).copy()
    plt.close(fig)
    image_buffer.close()
    return image


def translate_and_visualize(text, layer_idx, head_idx):
    if model is None or device is None or tokenizer_en is None or tokenizer_it is None:
        raise RuntimeError("The model and tokenizers must be initialized before translation.")

    tokens = tokenizer_en.encode(text).ids
    if len(tokens) + 2 > 128:
        raise ValueError("Input must contain no more than 126 English tokens.")

    src = torch.tensor([2, *tokens, 3], dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (src != 0).unsqueeze(0).unsqueeze(0).int().to(device)
    attention_by_layer = {}

    def capture_attention(layer_idx):
        def hook(*hook_args):
            attention_by_layer[layer_idx] = hook_args[2][1].detach()

        return hook

    hooks = [
        layer.cross_attn.register_forward_hook(capture_attention(index))
        for index, layer in enumerate(model.decoder.layers)
    ]
    try:
        translated_tokens = greedy_decode(
            model, src, src_mask, max_len=50, bos_idx=2, eos_idx=3, device=device
        )
        decoder_input = translated_tokens.unsqueeze(0)
        tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        with torch.no_grad():
            enc_output = model.encode(src, src_mask)
            model.decode(decoder_input, enc_output, src_mask, tgt_mask)
    finally:
        for hook in hooks:
            hook.remove()

    encoder_tokens = [
        tokenizer_en.id_to_token(token_id) or f"<{token_id}>"
        for token_id in src.squeeze(0).tolist()
    ]
    decoder_tokens = [
        tokenizer_it.id_to_token(token_id) or f"<{token_id}>"
        for token_id in translated_tokens.tolist()
    ]
    attention_weights = torch.stack([
        attention_by_layer[index][0]
        for index in range(len(model.decoder.layers))
    ])
    attention_image = visualize_attention(
        encoder_tokens,
        decoder_tokens,
        attention_weights,
        layer_idx=int(layer_idx),
        head_idx=int(head_idx),
    )
    return tokenizer_it.decode(translated_tokens.tolist()), attention_image


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
        max_len=128
    ).to(device)
    
    try:
        state_dict = torch.load("transformer_en_it.pt", map_location=device)
        state_dict.pop("encoder.pos_encoding.pe", None)
        state_dict.pop("decoder.pos_encoding.pe", None)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print("Model weights file 'transformer_en_it.pt' not found. Run train.py first!")
        exit()

    # 3. Test Translation on a Sample Sentence
    demo = gr.Interface(
        fn=translate_and_visualize,
        inputs=[
            gr.Textbox(label="English input", lines=3),
            gr.Slider(
                minimum=0,
                maximum=len(model.decoder.layers) - 1,
                step=1,
                label="Decoder layer",
            ),
            gr.Slider(
                minimum=0,
                maximum=model.decoder.layers[0].cross_attn.num_heads - 1,
                step=1,
                label="Attention head",
            ),
        ],
        outputs=[
            gr.Textbox(label="Italian translation"),
            gr.Image(type="pil", label="Cross-attention map"),
        ],
    )
    demo.launch(share=True)