
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer

def inspect_data(data_path, tokenizer_path):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = [json.loads(line) for line in lines]
    print(f"Total samples: {len(data)}")
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    code_lengths = []
    summary_lengths = []
    unk_counts = 0
    total_tokens = 0
    
    print("Analyzing samples...")
    # Analyze a subset if too large, but 200k is fine to do quickly
    for item in tqdm(data):
        code = item['code']
        summary = item['summary']
        
        # Tokenize (without adding special tokens for raw length check)
        code_ids = tokenizer.encode(code).ids
        summary_ids = tokenizer.encode(summary).ids
        
        code_lengths.append(len(code_ids))
        summary_lengths.append(len(summary_ids))
        
        # Check for UNK ([UNK] is usually ID 1 or check tokenizer)
        unk_id = tokenizer.token_to_id("[UNK]")
        unk_counts += code_ids.count(unk_id) + summary_ids.count(unk_id)
        total_tokens += len(code_ids) + len(summary_ids)

    print("\n=== Data Statistics ===")
    print(f"Code Tokens (Avg): {np.mean(code_lengths):.2f} +/- {np.std(code_lengths):.2f}")
    print(f"Code Tokens (Max): {np.max(code_lengths)}")
    print(f"Summary Tokens (Avg): {np.mean(summary_lengths):.2f} +/- {np.std(summary_lengths):.2f}")
    print(f"Summary Tokens (Max): {np.max(summary_lengths)}")
    print(f"UNK Token Rate: {unk_counts / total_tokens:.4%}")
    
    print("\n=== Random Examples ===")
    for i in range(5):
        sample = random.choice(data)
        print(f"\n[Example {i+1}]")
        print("Code (snippet):")
        print(sample['code'][:200] + "..." if len(sample['code']) > 200 else sample['code'])
        print("-" * 20)
        print("Summary:")
        print(sample['summary'])
        print("=" * 40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/processed/codesearchnet_clean/train.jsonl")
    parser.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    args = parser.parse_args()
    
    inspect_data(args.data_file, args.tokenizer_path)

if __name__ == "__main__":
    main()
