import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import numpy as np
import json
import gc

logging.set_verbosity_error()

# Config (PubMedBERT)
EPOCHS = 5
LEARNING_RATE = 1e-5
PHYSICAL_BATCH = 8
ACCUM_STEPS = 4
PATIENCE = 2
NUM_FOLDS = 10

# Accuracy function (top-k)
def get_accuracy_topk(logits, labels, k=1):
    _, top_k_indices = torch.topk(logits, k, dim=1)
    labels = labels.view(-1, 1).expand_as(top_k_indices)
    correct = (top_k_indices == labels).sum().item()
    return correct / labels.size(0)

# Main entrypoint
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # TF32 Boost (Ampere GPUs like RTX 3060)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading triage tensors...")
    try:
        data = torch.load('../BERT_L/triage_tensors.pt', weights_only=False)
    except FileNotFoundError:
        print("triage_tensors.pt not found.")
        return

    all_input_ids = data['input_ids']
    all_attention_masks = data['attention_mask']
    all_labels = data['labels'].long()

    print(f"Total Samples: {len(all_labels)}")

    graph_data = {"batch_loss": [], "batch_acc": [], "batch_acc_top2": []}

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_top1_scores = []
    fold_top2_scores = []
    best_overall_top1 = 0.0

    # 10-fold cross-validation loop
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(all_input_ids, all_labels)):

        print("\n" + "="*50)
        print(f"Fold {fold_i+1} / {NUM_FOLDS}")
        print("="*50)

        train_data = TensorDataset(
            all_input_ids[train_idx],
            all_attention_masks[train_idx],
            all_labels[train_idx]
        )

        val_data = TensorDataset(
            all_input_ids[val_idx],
            all_attention_masks[val_idx],
            all_labels[val_idx]
        )

        train_loader = DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            batch_size=PHYSICAL_BATCH,
            pin_memory=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_data,
            sampler=SequentialSampler(val_data),
            batch_size=PHYSICAL_BATCH * 2,
            pin_memory=True,
            num_workers=0
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            num_labels=20,
            use_safetensors=True
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            fused=True
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=1
        )

        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float('inf')
        best_fold_top1 = 0.0
        best_fold_top2 = 0.0
        early_stop_counter = 0

        # Epoch loop
        for epoch_i in range(EPOCHS):

            model.train()
            total_train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):

                b_ids = batch[0].to(device, non_blocking=True)
                b_mask = batch[1].to(device, non_blocking=True)
                b_labels = batch[2].to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(
                        b_ids,
                        attention_mask=b_mask,
                        labels=b_labels
                    )
                    loss = outputs.loss / ACCUM_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                actual_loss = loss.item() * ACCUM_STEPS
                logits = outputs.logits.detach()

                graph_data["batch_loss"].append(actual_loss)
                graph_data["batch_acc"].append(get_accuracy_topk(logits, b_labels, k=1))
                graph_data["batch_acc_top2"].append(get_accuracy_topk(logits, b_labels, k=2))

                total_train_loss += actual_loss

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation pass
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_true = []

            with torch.no_grad():
                for batch in val_loader:

                    b_ids = batch[0].to(device, non_blocking=True)
                    b_mask = batch[1].to(device, non_blocking=True)
                    b_labels = batch[2].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):
                        outputs = model(
                            b_ids,
                            attention_mask=b_mask,
                            labels=b_labels
                        )

                    total_val_loss += outputs.loss.item()
                    all_preds.append(outputs.logits.detach().cpu())
                    all_true.append(b_labels.cpu())

            avg_val_loss = total_val_loss / len(val_loader)

            all_preds = torch.cat(all_preds, dim=0)
            all_true = torch.cat(all_true, dim=0)

            top1 = get_accuracy_topk(all_preds, all_true, k=1)
            top2 = get_accuracy_topk(all_preds, all_true, k=2)

            print(
                f"Epoch {epoch_i+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Top-1: {top1:.2%} | Top-2: {top2:.2%}"
            )

            scheduler.step(avg_val_loss)

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_fold_top1 = top1
                best_fold_top2 = top2
                early_stop_counter = 0

                if top1 > best_overall_top1:
                    best_overall_top1 = top1
                    # Save best overall model
                    torch.save(model.state_dict(), "pubmedbert_best_model.pth")
                    print(f"[New Best Model Saved | Top-1: {top1:.2%}]")

            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print("[Early Stopping Triggered]")
                    break

        print(f"Fold {fold_i+1} Best | Top-1: {best_fold_top1:.2%} | Top-2: {best_fold_top2:.2%}")

        fold_top1_scores.append(best_fold_top1)
        fold_top2_scores.append(best_fold_top2)

        # VRAM cleanup between folds
        del model, optimizer, scheduler, scaler
        del train_loader, val_loader, train_data, val_data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\n" + "="*50)
    print("FINAL 10-FOLD RESULTS")
    print(f"Avg Top-1: {np.mean(fold_top1_scores):.4f} ± {np.std(fold_top1_scores):.4f}")
    print(f"Avg Top-2: {np.mean(fold_top2_scores):.4f} ± {np.std(fold_top2_scores):.4f}")
    print("="*50)

    with open("pubmedbert_training_graph_data.json", "w") as f:
        json.dump(graph_data, f)


if __name__ == "__main__":
    main()