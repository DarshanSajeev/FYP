import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix

# Setup and Labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_map = {
    0: "Respiration", 1: "Gastroenterology", 2: "Cardiology",
    3: "Neurology", 4: "Dermatology", 5: "Orthopedics",
    6: "Oncology", 7: "Endocrinology", 8: "Nephrology",
    9: "Hepatology", 10: "Psychiatry", 11: "Ophthalmology",
    12: "Otolaryngology", 13: "Urology", 14: "Gynecology",
    15: "Pediatrics", 16: "Rheumatology", 17: "Immunology",
    18: "Infectious Diseases", 19: "Other"
}
label_names = [label_map[i] for i in range(20)]

def generate_normalized_matrix():
    print(f"Loading model on {device}...")
    
    # Load the Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=20, output_attentions=False, output_hidden_states=False
    )
    model.load_state_dict(torch.load("bert_model_fold_0.pth", map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    # Load the Data
    print("Loading triage tensors...")
    try:
        data = torch.load('triage_tensors.pt', map_location=device, weights_only=False)
    except FileNotFoundError:
        print("Error: Could not find 'triage_tensors.pt' in this folder.")
        return

    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    # Batch size 32 so we don't run out of VRAM on your RTX 3060
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    true_labels = []
    predicted_labels = []

    # Run Predictions
    print("Running predictions (this will take a minute)...")
    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        
        true_labels.extend(b_labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())
        
        if (step + 1) % 20 == 0:
            print(f"  Processed {step * 32} patients...")

    print("\nPredictions complete! Generating normalized graph...")

    # Plot the Normalized Confusion Matrix
    cm_normalized = confusion_matrix(true_labels, predicted_labels, normalize='true')

    plt.figure(figsize=(16, 12))
    plt.title('Normalized Confusion Matrix: BERT_L (20-Class)', fontsize=18, fontweight='bold')

    # Plot using Seaborn (fmt='.2f' gives the 0.95 decimal format)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=label_names, 
                yticklabels=label_names,
                cbar_kws={'label': 'Proportion of Correct Predictions'})

    plt.ylabel('True Clinical Department', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Clinical Department', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix_BERT_L.png', dpi=300)
    print("Success! Saved to 'normalized_confusion_matrix_BERT_L.png'")
    plt.show()

if __name__ == "__main__":
    generate_normalized_matrix()