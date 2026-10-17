import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """
    Custom PyTorch Dataset for handling English-Italian parallel translations.
    """
    def __init__(self, ds, tokenizer_en, tokenizer_it, src_lang, tgt_lang, max_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_en = tokenizer_en
        self.tokenizer_it = tokenizer_it
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

        # Define special token IDs based on our word-level tokenizer setup
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize text into IDs
        src_tokens = self.tokenizer_en.encode(src_text).ids
        tgt_tokens = self.tokenizer_it.encode(tgt_text).ids

        # Calculate how much padding is needed to match max_len
        src_num_padding = self.max_len - len(src_tokens) - 2  # Accounting for BOS and EOS
        tgt_num_padding = self.max_len - len(tgt_tokens) - 1  # Accounting for BOS or EOS

        if src_num_padding < 0 or tgt_num_padding < 0:
            raise ValueError("Sentence is too long for the specified max_len!")

        # Build Encoder Input: [BOS] + tokens + [EOS] + [PAD]...
        encoder_input = torch.cat([
            torch.tensor([self.bos_token_id], dtype=torch.int64),
            torch.tensor(src_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * src_num_padding, dtype=torch.int64),
        ])

        # Build Decoder Input (shifted right for teacher forcing): [BOS] + tokens + [PAD]...
        decoder_input = torch.cat([
            torch.tensor([self.bos_token_id], dtype=torch.int64),
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * tgt_num_padding, dtype=torch.int64),
        ])

        # Build Label (what the decoder is expected to output): tokens + [EOS] + [PAD]...
        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * tgt_num_padding, dtype=torch.int64),
        ])

        return {
            "encoder_input": encoder_input,  # (max_len)
            "decoder_input": decoder_input,  # (max_len)
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, max_len)
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, max_len, max_len)
            "label": label,  # (max_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    """
    Creates a lower-triangular mask to prevent the decoder from looking into the future.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0