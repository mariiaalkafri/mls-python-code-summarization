# scripts/generate.py
import os
import sys
import argparse
import math
import torch
from tokenizers import Tokenizer

# Ensure src can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_model import TransformerSeq2Seq


def load_tokenizer(tokenizer_path: str):
    tok = Tokenizer.from_file(tokenizer_path)
    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    unk_id = tok.token_to_id("[UNK]")

    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError(
            "Tokenizer is missing required special tokens. "
            "Need: [PAD], [BOS], [EOS] (and ideally [UNK])."
        )
    return tok, pad_id, bos_id, eos_id, unk_id


def encode_source(tok: Tokenizer, code: str, bos_id: int, eos_id: int, max_src_len: int):
    ids = tok.encode(code).ids
    # Reserve 2 tokens for BOS/EOS
    ids = ids[: max(0, max_src_len - 2)]
    src_ids = [bos_id] + ids + [eos_id]
    return src_ids


def pad_1d(ids, pad_id, max_len):
    ids = ids[:max_len]
    mask = [1] * len(ids)  # 1=real
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = ids + [pad_id] * pad_n
        mask = mask + [0] * pad_n  # 0=pad
    return ids, mask


def decode_tokens(tok: Tokenizer, ids, pad_id, bos_id, eos_id):
    # Remove specials
    ids = [i for i in ids if i not in (pad_id, bos_id, eos_id)]
    text = tok.decode(ids).strip()
    return text


def load_model(ckpt_path: str, vocab_size: int, pad_id: int, device: str):
    # Keep these hyperparams consistent with training (edit if your train.py differs)
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=pad_id,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # support raw state_dict too
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate_summary(
    model: TransformerSeq2Seq,
    tok: Tokenizer,
    code: str,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    device: str,
    max_src_len: int = 256,
    max_new_tokens: int = 64,
):
    src_ids = encode_source(tok, code, bos_id, eos_id, max_src_len=max_src_len)
    src_ids, src_mask = pad_1d(src_ids, pad_id, max_src_len)

    src_ids_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask_t = torch.tensor([src_mask], dtype=torch.long, device=device)  # 1=real,0=pad

    out_seqs = model.generate(
        src_ids=src_ids_t,
        src_mask=src_mask_t,
        max_len=max_new_tokens,
        bos_id=bos_id,
        eos_id=eos_id,
        min_len=3,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
    )

    # model.generate returns List[List[int]]
    out_ids = out_seqs[0]
    return decode_tokens(tok, out_ids, pad_id, bos_id, eos_id)


def read_code_from_stdin():
    print("Paste your Python code. End input with a line containing only:  EOF")
    lines = []
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate summary for Python code.")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt / last.pt).")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a .py file to summarize.")
    parser.add_argument("--code", type=str, default=None, help="Inline code string to summarize.")
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Tokenizer:", args.tokenizer)
    print("Checkpoint:", args.ckpt)

    tok, pad_id, bos_id, eos_id, _ = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = load_model(args.ckpt, vocab_size=vocab_size, pad_id=pad_id, device=device)

    # Get code
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        code = read_code_from_stdin()

    pred = generate_summary(
        model=model,
        tok=tok,
        code=code,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        max_src_len=args.max_src_len,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== SUMMARY ===")
    print(pred)


if __name__ == "__main__":
    main()
