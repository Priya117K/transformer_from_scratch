import os
import re
import time

import gradio as gr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from models.transformer import Transformer
from data.dataset import causal_mask

MAX_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 50
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

model: Transformer | None = None
device: torch.device | None = None
tokenizer_en: Tokenizer | None = None
tokenizer_it: Tokenizer | None = None


def _adjust_logits(logits, generated_tokens, temperature, repetition_penalty):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if repetition_penalty < 1:
        raise ValueError("Repetition penalty must be at least 1.0.")

    logits = logits.clone() / temperature
    logits[:, PAD_TOKEN_ID] = -torch.inf
    logits[:, BOS_TOKEN_ID] = -torch.inf
    repeated_tokens = generated_tokens.unique()
    repeated_tokens = repeated_tokens[
        (repeated_tokens != PAD_TOKEN_ID)
        & (repeated_tokens != BOS_TOKEN_ID)
        & (repeated_tokens != EOS_TOKEN_ID)
    ]
    if repeated_tokens.numel():
        repeated_logits = logits[:, repeated_tokens]
        logits[:, repeated_tokens] = torch.where(
            repeated_logits < 0,
            repeated_logits * repetition_penalty,
            repeated_logits / repetition_penalty,
        )
    return logits


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    bos_idx,
    eos_idx,
    device,
    repetition_penalty=1.2,
    temperature=1.0,
):
    """Generate target tokens autoregressively using greedy selection."""
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if repetition_penalty < 1:
        raise ValueError("Repetition penalty must be at least 1.0.")
    model.eval()
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        decoder_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            dec_output = model.decode(decoder_input, enc_output, src_mask, tgt_mask)
            logits = _adjust_logits(
                model.project(dec_output[:, -1]),
                decoder_input,
                temperature,
                repetition_penalty,
            )
            _, next_word = torch.max(logits, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_word.item()]], dtype=torch.long, device=device)], dim=1
            )
            if next_word.item() == eos_idx:
                break
    return decoder_input.squeeze(0)


def beam_search_decode(
    model,
    src,
    src_mask,
    max_len,
    bos_idx,
    eos_idx,
    device,
    repetition_penalty=1.2,
    temperature=1.0,
    beam_width=4,
):
    """Generate target tokens by retaining the highest-scoring beam candidates."""
    if beam_width < 1:
        raise ValueError("Beam width must be at least 1.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if repetition_penalty < 1:
        raise ValueError("Repetition penalty must be at least 1.0.")

    model.eval()
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        beams = [([bos_idx], 0.0)]
        for _ in range(max_len - 1):
            candidates = []
            for tokens, score in beams:
                if tokens[-1] == eos_idx:
                    candidates.append((tokens, score))
                    continue
                decoder_input = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
                dec_output = model.decode(decoder_input, enc_output, src_mask, tgt_mask)
                logits = _adjust_logits(
                    model.project(dec_output[:, -1]),
                    decoder_input,
                    temperature,
                    repetition_penalty,
                )
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                token_scores, token_ids = torch.topk(log_probs, min(beam_width, log_probs.numel()))
                candidates.extend(
                    (tokens + [token_id.item()], score + token_score.item())
                    for token_score, token_id in zip(token_scores, token_ids)
                )

            beams = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)[:beam_width]
            if all(tokens[-1] == eos_idx for tokens, _ in beams):
                break

    best_tokens, _ = max(beams, key=lambda candidate: candidate[1])
    return torch.tensor(best_tokens, dtype=torch.long, device=device)


def visualize_attention(encoder_tokens, decoder_tokens, attention_weights, layer_idx=0, head_idx=0):
    """Create a Matplotlib cross-attention heatmap for the selected layer and head."""
    fig, ax = plt.subplots(figsize=(9, 6))
    attn = attention_weights[layer_idx][head_idx].detach().cpu().numpy()
    sns.heatmap(
        attn[:len(decoder_tokens), :len(encoder_tokens)],
        xticklabels=encoder_tokens,
        yticklabels=decoder_tokens,
        cmap="viridis",
        annot=len(encoder_tokens) <= 30 and len(decoder_tokens) <= 30,
        ax=ax
    )
    ax.set_xlabel("English input tokens")
    ax.set_ylabel("Italian output tokens")
    ax.set_title(f"Cross-attention · Layer {layer_idx + 1} · Head {head_idx + 1}")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    return fig


def translate_and_visualize(
    text,
    decoding_strategy,
    temperature,
    repetition_penalty,
    layer_idx,
    head_idx,
):
    if model is None or device is None or tokenizer_en is None or tokenizer_it is None:
        raise RuntimeError("The model and tokenizers must be initialized before translation.")
    if decoding_strategy not in {"Greedy", "Beam search"}:
        raise ValueError(f"Unsupported decoding strategy: {decoding_strategy}")

    cleaned_text = re.sub(r"\[(?:PAD|UNK|BOS|EOS)\]", " ", text or "", flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned_text:
        raise ValueError("Enter an English sentence to translate.")

    tokens = tokenizer_en.encode(cleaned_text).ids
    if len(tokens) + 2 > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Input is too long. Please use at most {MAX_SEQUENCE_LENGTH - 2} English tokens."
        )
    src = torch.tensor(
        [BOS_TOKEN_ID, *tokens, EOS_TOKEN_ID], dtype=torch.long
    ).unsqueeze(0).to(device)
    src_mask = (src != PAD_TOKEN_ID).unsqueeze(0).unsqueeze(0).int().to(device)
    attention_by_layer = {}

    def capture_attention(index):
        def hook(*hook_args):
            attention_by_layer[index] = hook_args[2][1].detach()

        return hook

    hooks = [
        layer.cross_attn.register_forward_hook(capture_attention(index))
        for index, layer in enumerate(model.decoder.layers)
    ]
    started_at = time.perf_counter()
    try:
        decode_fn = greedy_decode if decoding_strategy == "Greedy" else beam_search_decode
        translated_tokens = decode_fn(
            model,
            src,
            src_mask,
            max_len=MAX_GENERATION_LENGTH,
            bos_idx=BOS_TOKEN_ID,
            eos_idx=EOS_TOKEN_ID,
            device=device,
            repetition_penalty=float(repetition_penalty),
            temperature=float(temperature),
        )
        decoder_input = translated_tokens.unsqueeze(0)
        tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        with torch.no_grad():
            enc_output = model.encode(src, src_mask)
            model.decode(decoder_input, enc_output, src_mask, tgt_mask)
        elapsed = time.perf_counter() - started_at
    finally:
        for hook in hooks:
            hook.remove()

    special_token_ids = {PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID}
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
    attention_figure = visualize_attention(
        encoder_tokens,
        decoder_tokens,
        attention_weights,
        layer_idx=int(layer_idx),
        head_idx=int(head_idx),
    )
    output_ids = [
        token_id for token_id in translated_tokens.tolist()
        if token_id not in special_token_ids
    ]
    translation = tokenizer_it.decode(output_ids, skip_special_tokens=True)
    translation = re.sub(r"\s+([,.;:!?])", r"\1", translation)
    translation = re.sub(r"\s+", " ", translation).strip().rstrip(",;:")
    return translation, attention_figure, elapsed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_en = Tokenizer.from_file(os.environ.get("TOKENIZER_EN_PATH", "data/tokenizer_en.json"))
    tokenizer_it = Tokenizer.from_file(os.environ.get("TOKENIZER_IT_PATH", "data/tokenizer_it.json"))
    src_vocab_size = tokenizer_en.get_vocab_size()
    tgt_vocab_size = tokenizer_it.get_vocab_size()
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model=768,
        num_layers=8,
        num_heads=12,
        d_ff=3072,
        max_len=MAX_SEQUENCE_LENGTH,
    ).to(device)
    model_path = os.environ.get("MODEL_PATH", "transformer_en_it.pt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict.pop("encoder.pos_encoding.pe", None)
    state_dict.pop("decoder.pos_encoding.pe", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model weights loaded from {model_path} on {device}.", flush=True)

    with gr.Blocks(title="English–Italian Translator") as demo:
        gr.Markdown("# English → Italian\nTranslate a sentence and inspect its cross-attention.")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="English input",
                    placeholder="Type an English sentence…",
                    lines=5,
                )
                decoding_strategy = gr.Radio(
                    ["Greedy", "Beam search"],
                    value="Greedy",
                    label="Decoding strategy",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature (beam search)",
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.05,
                    label="Repetition penalty",
                )
                layer_idx = gr.Slider(
                    minimum=0,
                    maximum=len(model.decoder.layers) - 1,
                    value=0,
                    step=1,
                    label="Attention layer",
                )
                head_idx = gr.Slider(
                    minimum=0,
                    maximum=model.decoder.layers[0].cross_attn.num_heads - 1,
                    value=0,
                    step=1,
                    label="Attention head",
                )
                translate_button = gr.Button("Translate", variant="primary")
            with gr.Column():
                translation_output = gr.Textbox(label="Italian translation", lines=5)
                attention_output = gr.Plot(label="Cross-attention heatmap")
                latency_output = gr.Number(label="Execution latency (seconds)", precision=3)

     
