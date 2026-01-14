
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.dataset import CodeSummaryDataset, collate_fn

def main():
    data_path = "data/processed/codesearchnet_clean/train.jsonl"
    tokenizer_path = "data/tokenizer/tokenizer.json"
    
    if not os.path.exists(data_path) or not os.path.exists(tokenizer_path):
        print("Data or tokenizer not found. Skipping verification.")
        return

    print("Initializing Dataset...")
    dataset = CodeSummaryDataset(data_path, tokenizer_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Check one item
    item = dataset[0]
    print("Sample Item Keys:", item.keys())
    print("Source IDs:", item['source_ids'][:10], "...")
    
    # DataLoader
    pad_id = dataset.pad_id
    loader = DataLoader(dataset, batch_size=4, collate_fn=lambda b: collate_fn(b, pad_id))
    
    print("\nChecking DataLoader batch...")
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("Source Shape:", batch['source_ids'].shape)
    print("Target Shape:", batch['target_ids'].shape)
    
    # Decode back
    print("\nDecoding first sample in batch:")
    src_ids = batch['source_ids'][0]
    tgt_ids = batch['target_ids'][0]
    
    # Remove pad
    src_ids = src_ids[src_ids != pad_id]
    tgt_ids = tgt_ids[tgt_ids != pad_id]
    
    src_text = dataset.tokenizer.decode(src_ids.tolist())
    tgt_text = dataset.tokenizer.decode(tgt_ids.tolist())
    
    print(f"Source: {src_text[:100]}...")
    print(f"Target: {tgt_text}")
    print("\nVerification Successful.")

if __name__ == "__main__":
    main()
