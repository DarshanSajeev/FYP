import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_predictions(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            masks = batch[1].to(device)
            labels = batch[2]
            
            outputs = model(inputs, attention_mask=masks)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_logits), torch.cat(all_labels)

def calculate_topk_accuracy(logits, labels, k=2):
    # Top-k only makes sense if k < num_classes
    if logits.size(1) < k:
        return 1.0 # If we have fewer classes than k, accuracy is trivially 100%
        
    _, topk_preds = torch.topk(logits, k, dim=1)
    correct = (topk_preds == labels.view(-1, 1)).any(dim=1).sum().item()
    return correct / labels.size(0)

def plot_confusion_matrix(labels, preds, model_name, class_count):
    """Generates and saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({class_count} Classes)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    # Safe filename formatting
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    filename = f"confusion_matrix_{safe_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  -> Saved {filename}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================
    # 1. CONFIGURATION (Edit paths here)
    # ==========================================
    # Each entry defines a specific model and the data it was trained on.
    models_to_evaluate = [
        {
            "name": "PubMedBERT (20-Class)",
            "weights_path": "PubMedBERT/pubmedbert_model_fold_0.pth",
            "base_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "num_labels": 20,
            "data_path": "BERT_L/triage_tensors.pt"  # 20-class dataset
        },
        {
            "name": "BERT_L (20-Class)",
            "weights_path": "BERT_L/bert_model_fold_0.pth",
            "base_model": "bert-base-uncased",          # Standard BERT
            "num_labels": 20,
            "data_path": "BERT_L/triage_tensors.pt"  # 20-class dataset
        },
        {
            "name": "BERT_S (5-Class)",
            "weights_path": "BERT_S/bert_model_fold_0.pth",
            "base_model": "bert-base-uncased",          # Standard BERT
            "num_labels": 5,
            "data_path": "BERT_S/triage_tensors.pt"  # 5-class dataset
        }
    ]
    
    comparison_results = []

    # ==========================================
    # 2. EVALUATION LOOP
    # ==========================================
    for config in models_to_evaluate:
        name = config["name"]
        print(f"\n" + "="*40)
        print(f"Processing: {name}")
        print(f"="*40)
        
        # --- A. Load the Correct Data for this Model ---
        if not os.path.exists(config["data_path"]):
            print(f"  [SKIP] Data file not found: {config['data_path']}")
            continue
            
        print(f"  Loading data: {config['data_path']}...")
        data = torch.load(config["data_path"], weights_only=False)
        
        # Create a test slice (last 500 samples)
        # We ensure labels are Long tensors to avoid dtype errors
        test_dataset = torch.utils.data.TensorDataset(
            data['input_ids'][-500:], 
            data['attention_mask'][-500:], 
            data['labels'][-500:].long()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # --- B. Load the Model Architecture & Weights ---
        print(f"  Loading model: {config['base_model']} ({config['num_labels']} labels)...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                config["base_model"], 
                num_labels=config["num_labels"],
                use_safetensors=True 
            ).to(device)
            
            # Load your trained weights
            if not os.path.exists(config["weights_path"]):
                print(f"  [ERROR] Weights file not found: {config['weights_path']}")
                continue
                
            model.load_state_dict(torch.load(config["weights_path"], map_location=device, weights_only=True))
            
        except RuntimeError as e:
            print(f"  [CRITICAL ERROR] Shape mismatch or load failure for {name}.")
            print(f"  Details: {e}")
            continue

        # --- C. Predict & Evaluate ---
        logits, labels = get_predictions(model, test_loader, device)
        preds = torch.argmax(logits, dim=1)
        
        # Metrics
        top1 = accuracy_score(labels, preds)
        # Only calc Top-2 if we have enough classes
        top2 = calculate_topk_accuracy(logits, labels, k=2) 
        f1 = f1_score(labels, preds, average='macro')
        
        print(f"  -> Accuracy: {top1:.2%}")
        
        plot_confusion_matrix(labels, preds, name, config["num_labels"])
        
        comparison_results.append({
            "Model": name,
            "Classes": config["num_labels"],
            "Top-1 Acc": top1,
            "Top-2 Acc": top2,
            "Macro F1": f1
        })

    if not comparison_results:
        print("\nNo models were successfully evaluated.")
        return

    # ==========================================
    # 3. VISUALIZATION
    # ==========================================
    df = pd.DataFrame(comparison_results)
    print("\n--- Final Model Comparison ---")
    print(df.to_string(index=False))

    # We reshape the data for plotting
    df_melted = df.melt(id_vars=["Model", "Classes"], var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("model_comparison_chart.png")
    print("\nComparison chart saved as 'model_comparison_chart.png'")

if __name__ == "__main__":
    main()