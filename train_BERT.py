import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import numpy as np
import json

#Accuracy calculation function
# k=1 for top-1 accuracy, k=2 for top-2 accuracy
# logics are the outputs of BERT 
# labels are the correct answer BERT predicted
def get_accuracy_topk(logits, labels, k=1):
    # Get the top k predictions
    # torch.topk returns gets the highest k values and their indices along a specified dimension
    _, top_k_indices = torch.topk(logits, k, dim=1)
    labels = labels.view(-1, 1).expand_as(top_k_indices)
    correct = (top_k_indices == labels).sum().item()
    return correct / labels.size(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading 'triage_tensors.pt'...")
    try:
        data = torch.load('triage_tensors.pt')
    except FileNotFoundError:
        print("Error: 'triage_tensors.pt' not found.")
        return

    all_input_ids = data['input_ids']
    all_attention_masks = data['attention_mask']
    all_labels = data['labels']

    print(f"Total Samples: {len(all_labels)}")

    N_FOLDS = 10          
    EPOCHS = 5            
    LEARNING_RATE = 1e-5  
    BATCH_SIZE = 32
    PATIENCE = 2          

    graph_data = {
        "batch_loss": [],
        "batch_acc": [],
        "batch_acc_top2": []
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_top1_scores = []
    fold_top2_scores = []

    print(f"\n--- STARTING {N_FOLDS}-FOLD CROSS VALIDATION ---")

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(all_input_ids, all_labels)):
        print(f"\n================ Fold {fold_i + 1} / {N_FOLDS} ================")
        
        train_inputs = all_input_ids[train_idx]
        train_masks = all_attention_masks[train_idx]
        train_labels = all_labels[train_idx]

        val_inputs = all_input_ids[val_idx]
        val_masks = all_attention_masks[val_idx]
        val_labels = all_labels[val_idx]

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=5,
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

        best_val_loss = float('inf')
        best_fold_top1 = 0.0
        best_fold_top2 = 0.0
        early_stopping_counter = 0

        for epoch_i in range(EPOCHS):
            model.train()
            total_train_loss = 0

            for batch in train_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = result.loss
                
                logits = result.logits.detach()
                acc_1 = get_accuracy_topk(logits, b_labels, k=1)
                acc_2 = get_accuracy_topk(logits, b_labels, k=2)
                
                graph_data["batch_loss"].append(loss.item())
                graph_data["batch_acc"].append(acc_1)
                graph_data["batch_acc_top2"].append(acc_2)

                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)

            model.eval()
            total_val_loss = 0
            all_preds = []
            all_true = []

            for batch in val_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                
                loss = result.loss
                total_val_loss += loss.item()
                
                logits = result.logits.detach().cpu()
                label_ids = b_labels.to('cpu')
                all_preds.append(logits)
                all_true.append(label_ids)

            avg_val_loss = total_val_loss / len(val_dataloader)

            all_preds = torch.cat(all_preds, dim=0)
            all_true = torch.cat(all_true, dim=0)
            top1_acc = get_accuracy_topk(all_preds, all_true, k=1)
            top2_acc = get_accuracy_topk(all_preds, all_true, k=2)

            print(f"  Epoch {epoch_i+1}: Val Loss: {avg_val_loss:.4f} | Top-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_fold_top1 = top1_acc
                best_fold_top2 = top2_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= PATIENCE:
                print("  [Early Stopping] Validation loss stopped improving.")
                break 
        
        print(f"  -> Fold {fold_i+1} Best Scores: Top-1: {best_fold_top1:.2%} | Top-2: {best_fold_top2:.2%}")
        fold_top1_scores.append(best_fold_top1)
        fold_top2_scores.append(best_fold_top2)

    print("\n" + "="*40)
    print("FINAL 10-FOLD CROSS VALIDATION RESULTS")
    print(f"Average Top-1 Accuracy: {np.mean(fold_top1_scores):.4f}")
    print(f"Average Top-2 Accuracy: {np.mean(fold_top2_scores):.4f}")
    
    print("Saving graph data to 'training_graph_data.json'...")
    with open('training_graph_data.json', 'w') as f:
        json.dump(graph_data, f)
    print("="*40)

if __name__ == "__main__":
    main()