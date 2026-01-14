
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        # batch_first=True => tensors are [B, T, E]
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device):
        # True means "block"
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src_ids, src_mask, tgt_in):
        """
        src_ids:  [B,S]
        src_mask: [B,S]  (1=real, 0=pad) -> needs inversion for torch.nn.Transformer if passing boolean mask
        tgt_in:   [B,T]
        returns logits: [B,T,V]
        """
        device = src_ids.device

        src = self.src_embedding(src_ids) * math.sqrt(self.d_model)  # [B,S,E]
        tgt = self.tgt_embedding(tgt_in) * math.sqrt(self.d_model)   # [B,T,E]
        src = self.pos(src)
        tgt = self.pos(tgt)

        # Transformer expects True for PADDING, False for REAL
        # src_mask is 1 for real, 0 for pad. So we need src_mask == 0
        src_key_padding_mask = (src_mask == 0)          # [B,S]
        tgt_key_padding_mask = (tgt_in == self.pad_id)  # [B,T]

        T = tgt_in.size(1)
        tgt_mask = self._causal_mask(T, device=device)  # [T,T]

        out = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )  # [B,T,E]

        logits = self.out(out)  # [B,T,V]
        return logits

    @staticmethod
    def _get_ngrams(tokens, n: int):
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @torch.no_grad()
    def generate(
        self,
        src_ids,
        src_mask,
        max_len: int = 64,
        bos_id: int = None,
        eos_id: int = None,
        min_len: int = 3,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
    ):
        """
        Greedy decoding + anti-repetition controls.
        Returns: List[List[int]] (each includes BOS then generated tokens)
        """
        if bos_id is None or eos_id is None:
            raise ValueError("bos_id and eos_id must be provided")

        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            # src_mask needs to be inverted for forward pass internally if used, 
            # but forward() above handles it.
            logits = self.forward(src_ids, src_mask, ys)   # [B,t,V]
            next_logits = logits[:, -1, :]                 # [B,V]

            # repetition penalty
            if repetition_penalty is not None and repetition_penalty > 1.0:
                for b in range(B):
                    used = ys[b].tolist()
                    next_logits[b, used] = next_logits[b, used] / repetition_penalty

            # no-repeat ngram
            if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1:
                n = no_repeat_ngram_size
                V = next_logits.size(-1)
                for b in range(B):
                    if finished[b]:
                        continue
                    prev = ys[b].tolist()
                    if len(prev) >= n - 1:
                        prefix = tuple(prev[-(n - 1):])
                        ngrams = self._get_ngrams(prev, n)
                        banned = []
                        for cand in range(V):
                            if prefix + (cand,) in ngrams:
                                banned.append(cand)
                        if banned:
                            next_logits[b, banned] = torch.finfo(next_logits.dtype).min

            # avoid too-early eos
            if step < min_len:
                next_logits[:, eos_id] = torch.finfo(next_logits.dtype).min

            next_token = torch.argmax(next_logits, dim=-1)

            next_token = torch.where(
                finished,
                torch.tensor(eos_id, device=device),
                next_token
            )

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == eos_id)

            if torch.all(finished):
                break

        return [seq.tolist() for seq in ys]
