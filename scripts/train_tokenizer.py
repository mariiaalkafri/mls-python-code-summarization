
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_tokenizer(input_file, output_file, vocab_size=30000):
    print(f"Training tokenizer on {input_file}...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=True
    )
    
    # Data iterator
    def batch_iterator():
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['code']
                yield data['summary']
                
    # Train
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Post-processor (optional but good for template)
    # RoBERTa-like processing: 
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    print(f"Saving tokenizer to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tokenizer.save(output_file)
    print("Tokenizer saved.")

def main():
    input_path = "data/processed/codesearchnet_clean/train.jsonl"
    output_path = "data/tokenizer/tokenizer.json"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Run preprocess_data.py first.")
        return

    train_tokenizer(input_path, output_path)

if __name__ == "__main__":
    main()
