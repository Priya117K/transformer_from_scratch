from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

def get_all_sentences(dataset, language):
    """Generator to yield sentences for training the tokenizer."""
    for item in dataset:
        yield item["translation"][language]

def build_tokenizer(dataset, lang_code, vocab_size=32000):
    print(f"Training custom word-level tokenizer for {lang_code}...")
    
    # 1. Initialize a WordLevel tokenizer with an [UNK] (unknown) token
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    
    # 2. Use whitespace pre-tokenization to split words by spaces/punctuation
    tokenizer.pre_tokenizer = Whitespace()
    
    # 3. Set up the trainer with special tokens
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    
    # 4. Train the tokenizer on the dataset sentences
    tokenizer.train_from_iterator(get_all_sentences(dataset, lang_code), trainer=trainer)
    
    return tokenizer

def validator_check(trainer):
    return trainer

if __name__ == "__main__":
    print("Loading OPUS Books dataset (en-it)...")
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']
    
    # Create directory for saving tokenizers if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Train English Tokenizer
    en_tokenizer = build_tokenizer(dataset, "en")
    en_tokenizer.save("data/tokenizer_en.json")
    
    # Train Italian Tokenizer
    it_tokenizer = build_tokenizer(dataset, "it")
    it_tokenizer.save("data/tokenizer_it.json")
    
    print("Tokenizers successfully trained and saved in the 'data/' folder! 🚀")