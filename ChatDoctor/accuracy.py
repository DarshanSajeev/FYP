import json
import numpy as np

def get_exact_accuracy(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get the arrays
    acc_1 = np.array(data['batch_acc'])
    acc_2 = np.array(data['batch_acc_top2'])
    
    # Calculate the exact mean across the entire training run
    final_top1 = np.mean(acc_1) * 100
    final_top2 = np.mean(acc_2) * 100
    
    print(f"--- RESULTS FOR {filepath} ---")
    print(f"Exact Top-1 Accuracy: {final_top1:.2f}%")
    print(f"Exact Top-2 Accuracy: {final_top2:.2f}%\n")

# Run it for BERT_S (Make sure the path is correct for your machine)
get_exact_accuracy('training_graph_data.json')