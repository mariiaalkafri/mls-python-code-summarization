import os
import sys
import random
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure src can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import JsonlCodeSummaryDataset, Collator
from src.transformer_model import TransformerSeq2Seq
from src.train_utils_transformer import train_model


def _bucket_index(value: int, boundaries: List[int]) -> int:
    for i, b in enumerate(boundaries):
        if value <= b:
            return i
    return len(boundaries)


def pick_2d_stratified_subset(
    examples: List[Dict],
    n: int,
    seed: int = 42,
    code_boundaries: List[int] = None,
    sum_boundaries: List[int] = None,
    code_buckets: int = 5,
    sum_buckets: int = 3,
) -> Tuple[List[Dict], List[Tuple[Tuple[int, int], int, int]]]:
    if n is None or n >= len(examples):
        return examples, []

    rng = random.Random(seed)

    if code_boundaries is None:
        code_boundaries = [200, 600, 1200, 2400]
    if sum_boundaries is None:
        sum_boundaries = [40, 90]

    cells = {(i, j): [] for i in range(code_buckets) for j in range(sum_buckets)}

    for ex in examples:
        code_len = len(ex.get("code", ""))
        sum_len = len(ex.get("summary", ""))
        cb = min(_bucket_index(code_len, code_boundaries), code_buckets - 1)
        sb = min(_bucket_index(sum_len, sum_boundaries), sum_buckets - 1)
        cells[(cb, sb)].append(ex)

    total = len(examples)
    quotas = {k: int(round(n * (len(v) / total))) for k, v in cells.items()}

    qsum = sum(quotas.values())
    if qsum != n:
        keys_sorted = sorted(cells.keys(), key=lambda k: len(cells[k]), reverse=True)
        diff = n - qsum
        idx = 0
        while diff != 0 and idx < len(keys_sorted) * 10:
            k = keys_sorted[idx % len(keys_sorted)]
            if diff > 0:
                quotas[k] += 1
                diff -= 1
            else:
                if quotas[k] > 0:
                    quotas[k] -= 1
                    diff += 1
            idx += 1

    subset = []
    for cell_key, cell_list in cells.items():
        q = quotas[cell_key]
        if q <= 0:
            continue
        if q >= len(cell_list):
            subset.extend(cell_list)
        else:
            subset.extend(rng.sample(cell_list, q))

    rng.shuffle(subset)
    subset = subset[:n]
    return subset, []


def main():
    # --- Config ---
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/codesearchnet_clean/train.jsonl"
    val_path   = "data/processed/codesearchnet_clean/train.jsonl"

    # Auto-detect validation file
    possible_val = "data/processed/codesearchnet_clean/validation.jsonl"
    if os.path.exists(possible_val):
        val_path = possible_val
    elif os.path.exists("data/processed/codesearchnet_clean/test.jsonl"):
        val_path = "data/processed/codesearchnet_clean/test.jsonl"

    batch_size = 32
    max_src_len = 256
    max_tgt_len = 64

    epochs_total = 60
    lr = 1e-4
    weight_decay = 0.01
    clip_grad = 1.0
    log_every = 200

    SUBSET_TRAIN = 50_000
    SUBSET_VAL = 8_000
    SEED = 42

    # --- GOOGLE DRIVE PATH ---
    SAVE_DIR = "/content/drive/MyDrive/ml-python-code-summarization/models_transformer_fromscratch"
    RESUME_PATH = f"{SAVE_DIR}/last.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")
    print(f"Resume: {RESUME_PATH}")
    print(f"Save dir: {SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    pad_id = collator.pad_id
    vocab_size = collator.tokenizer.get_vocab_size()

    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)
    print(f"Full train examples: {len(train_dataset)}")
    print(f"Full val examples:   {len(val_dataset)}")

    # Stratified Sampling
    print("Selecting stratified subsets...")
    train_subset, _ = pick_2d_stratified_subset(train_dataset.examples, min(SUBSET_TRAIN, len(train_dataset)), seed=SEED)
    val_subset, _ = pick_2d_stratified_subset(val_dataset.examples, min(SUBSET_VAL, len(val_dataset)), seed=SEED)
    train_dataset.examples = train_subset
    val_dataset.examples = val_subset
    print(f"Subset selected -> train={len(train_dataset)}  val={len(val_dataset)}")

    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,  # Safe on Colab (Linux) for speed
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,  # Safe on Colab (Linux) for speed
        pin_memory=True
    )

    print("Initializing Transformer model...")
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=pad_id
    ).to(device)

    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pad_id=pad_id,
        epochs_total=epochs_total,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=SAVE_DIR,
        resume_path=RESUME_PATH,
        log_every=log_every,
        clip_grad=clip_grad
    )


if __name__ == "__main__":
    main()
