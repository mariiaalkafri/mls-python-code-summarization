import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(path, model, optimizer, epoch, val_loss, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val": best_val,
    }
    torch.save(state, path)


def _get(batch, key):
    """Support dict batches and object batches."""
    if isinstance(batch, dict):
        return batch.get(key, None)
    return getattr(batch, key, None)


def run_epoch(
    model, dataloader, optimizer, criterion, device,
    train=True, clip_grad=1.0, log_every=200, pad_id=0,
    scaler=None
):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    start = time.time()

    # ✅ AMP works for "cuda" and "cuda:0"
    use_amp = (scaler is not None) and str(device).startswith("cuda")

    for i, batch in enumerate(dataloader):
        # Required keys
        src_ids  = _get(batch, "src_ids")
        src_mask = _get(batch, "src_mask")

        if src_ids is None or src_mask is None:
            raise KeyError(
                "Batch must contain 'src_ids' and 'src_mask'. "
                f"Got keys: {list(batch.keys()) if isinstance(batch, dict) else dir(batch)}"
            )

        src_ids  = src_ids.to(device, non_blocking=True)
        src_mask = src_mask.to(device, non_blocking=True)

        # Two possible target formats:
        tgt_ids = _get(batch, "tgt_ids")   # Format A
        tgt_in  = _get(batch, "tgt_in")    # Format B
        labels  = _get(batch, "labels")    # Format B

        if tgt_ids is not None:
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            tgt_in = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]
        else:
            if tgt_in is None or labels is None:
                raise KeyError(
                    "Batch must contain either 'tgt_ids' OR ('tgt_in' and 'labels'). "
                    f"Got keys: {list(batch.keys()) if isinstance(batch, dict) else dir(batch)}"
                )
            tgt_in  = tgt_in.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(src_ids, src_mask, tgt_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(src_ids, src_mask, tgt_in)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )

        total_loss += float(loss.item())

        # ✅ token accuracy ignoring PAD
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels.ne(pad_id)
            total_correct += (preds.eq(labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        if train and (i % log_every == 0):
            acc = 0.0 if total_tokens == 0 else (total_correct / total_tokens) * 100.0
            elapsed = time.time() - start
            print(f"  batch {i}/{len(dataloader)}  loss={loss.item():.4f}  acc={acc:.2f}%  time={elapsed:.2f}s")

    avg_loss = total_loss / max(1, len(dataloader))
    avg_acc = 0.0 if total_tokens == 0 else (total_correct / total_tokens) * 100.0
    return avg_loss, avg_acc


def train_model(
    model, train_loader, val_loader, device, pad_id,
    epochs_total=10, lr=3e-4, weight_decay=0.01,
    save_dir="models", log_every=200, clip_grad=1.0,
    resume_path=None
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 1
    best_val = float("inf")

    # ✅ Fix AMP deprecation + enable for cuda:0 too
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
        print(f"Checkpoint loaded. Last completed epoch: {start_epoch-1}. Best val so far: {best_val:.4f}")
    elif resume_path is not None:
        print(f"Resume checkpoint not found at: {resume_path} (starting fresh)")

    print(f"Will train from epoch {start_epoch} to {epochs_total}")
    if start_epoch > epochs_total:
        print("Nothing to do: start_epoch > epochs_total.")
        return

    for epoch in range(start_epoch, epochs_total + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, clip_grad=clip_grad, log_every=log_every, pad_id=pad_id,
            scaler=scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device,
            train=False, pad_id=pad_id,
            scaler=scaler
        )

        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} |   Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(f"{save_dir}/best.pt", model, optimizer, epoch, val_loss, best_val)
            print("  ✔ Saved new best model")

        save_checkpoint(f"{save_dir}/last.pt", model, optimizer, epoch, val_loss, best_val)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
