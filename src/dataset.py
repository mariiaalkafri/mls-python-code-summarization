
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer

class CodeSummaryDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_length=512):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
        # Get special token IDs
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item['code']
        summary = item['summary']
        
        # Encode
        # Source: [BOS] code [EOS]
        src_enc = self.tokenizer.encode(code).ids
        # Target: [BOS] summary [EOS]
        tgt_enc = self.tokenizer.encode(summary).ids
        
        # Truncate (leave room for BOS/EOS)
        if len(src_enc) > self.max_length - 2:
            src_enc = src_enc[:self.max_length - 2]
        if len(tgt_enc) > self.max_length - 2:
            tgt_enc = tgt_enc[:self.max_length - 2]
            
        src_ids = [self.bos_id] + src_enc + [self.eos_id]
        tgt_ids = [self.bos_id] + tgt_enc + [self.eos_id]
        
        return {
            "source_ids": torch.tensor(src_ids, dtype=torch.long),
            "target_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }

def collate_fn(batch, pad_id):
    source_ids = [item["source_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    
    # Pad
    source_padded = torch.nn.utils.rnn.pad_sequence(source_ids, batch_first=True, padding_value=pad_id)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
    
    # Attention masks (1 for real tokens, 0 for pad)
    source_mask = (source_padded != pad_id).long()
    target_mask = (target_padded != pad_id).long()
    
    return {
        "source_ids": source_padded,
        "source_mask": source_mask,
        "target_ids": target_padded,
        "target_mask": target_mask
    }
