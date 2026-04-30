import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from tqdm import tqdm

# ======================
# CONFIG
# ======================

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5
MAX_LEN = 256
NUM_FOLDS = 10
PATIENCE = 2  # early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD DATA
# ======================

with open("../Training Data/triage_berts_dataset.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

labels = np.array(labels)
num_labels = len(set(labels))

# ======================
# DATASET CLASS
# ======================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# ======================
# 10-FOLD CROSS VALIDATION
# ======================

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):

    print(f"\n========== Fold {fold+1}/{NUM_FOLDS} ==========")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ======================
    # MODEL
    # ======================

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        use_safetensors=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model = None

    # ======================
    # TRAINING LOOP
    # ======================

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            optimizer.zero_grad()

            # Mixed precision forward
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = criterion(outputs.logits, labels_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print("Train Loss:", avg_train_loss)

        # ======================
        # VALIDATION
        # ======================

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = criterion(outputs.logits, labels_batch)

                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print("Val Loss:", avg_val_loss)
        print("Val Accuracy:", val_acc)
        print("Val F1:", val_f1)

        scheduler.step(avg_val_loss)

        # ======================
        # EARLY STOPPING
        # ======================

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    fold_results.append(val_f1)

# ======================
# FINAL RESULTS
# ======================

print("\n========== FINAL RESULTS ==========")
print("Fold F1 Scores:", fold_results)
print("Mean F1:", np.mean(fold_results))
print("Std F1:", np.std(fold_results))
