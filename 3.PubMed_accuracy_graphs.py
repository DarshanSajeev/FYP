import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_average_graph():
    # Plot averaged training graphs from pubmedbert_training_graph_data.json
    filename = 'pubmedbert_training_graph_data.json'

    # Load JSON data
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return

    # Convert stored lists to numpy arrays
    raw_loss = np.array(data['batch_loss'])
    raw_acc = np.array(data['batch_acc'])
    raw_acc_top2 = np.array(data['batch_acc_top2'])

    # Merge the 10 folds by reshaping the long lists
    num_folds = 10
    
    # Calculate how many batches were in one fold
    fold_length = len(raw_loss) // num_folds
    
    # Reshape: (10 folds, 450 batches per fold)
    # This aligns Fold 1, Fold 2, etc., on top of each other
    reshaped_loss = raw_loss[:num_folds*fold_length].reshape(num_folds, fold_length)
    reshaped_acc = raw_acc[:num_folds*fold_length].reshape(num_folds, fold_length)
    reshaped_acc_top2 = raw_acc_top2[:num_folds*fold_length].reshape(num_folds, fold_length)

    # Compute per-batch averages across folds
    avg_loss = np.mean(reshaped_loss, axis=0)
    avg_acc = np.mean(reshaped_acc, axis=0)
    avg_acc_top2 = np.mean(reshaped_acc_top2, axis=0)

    batches = range(len(avg_loss))

    # Plot averaged curves
    plt.figure(figsize=(10, 6))
    
    # Plot Top-2 Accuracy (Blue)
    plt.plot(batches, avg_acc_top2, color='blue', label='Avg Top-2 Accuracy', linewidth=1.5, alpha=0.9)
    
    # Plot Top-1 Accuracy (Red)
    plt.plot(batches, avg_acc, color='red', label='Avg Accuracy', linewidth=1.5, alpha=0.9)
    
    # Plot Loss (Green)
    plt.plot(batches, avg_loss, color='#4CC417', label='Avg Loss', linewidth=1.5, alpha=0.9)

    # Styling and save
    plt.xlabel("Batch (Averaged across 10 Folds)", fontsize=12)
    plt.ylabel("Score / Loss", fontsize=12)
    plt.title("PubMedBERT 10-Fold Cross-Validation Average Performance", fontsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.legend(loc='center right', frameon=True, facecolor='white', framealpha=1)
    plt.tight_layout()
    save_path = os.path.join('..', 'Graphs', 'pubmedbert_smooth_averaged_graph.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print("Success! Saved smooth graph as 'pubmedbert_smooth_averaged_graph.png'")
    plt.show()

if __name__ == "__main__":
    plot_average_graph()