import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label mapping (matches training)
label_map = {
    0: "Flu",
    1: "Pregnancy", 
    2: "Sexual Intercourse", 
    3: "Period", 
    4: "Exercise"
}

id_to_label = label_map
label_to_id = {v: k for k, v in label_map.items()}

def predict_with_analysis():
    # Make predictions on test data and generate analysis visualizations
    print(f"Loading model on {device}...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False
    )

    state_dict = torch.load(
        "bert_model_fold_0.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.\n")

    # Load test data
    csv_path = os.path.join("Test Data", "test.csv") 
    
    print(f"Reading data from '{csv_path}'...")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'.")
        return
    
    try:
        df = pd.read_csv(csv_path, nrows=2000)
        print(f"Successfully loaded {len(df)} rows for testing.\n")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    predictions = []
    prediction_ids = []
    texts = []

    print("Running predictions...")
    
    for index, row in df.iterrows():
        # Extract text
        text = str(row['TEXT']) if 'TEXT' in row else str(row.iloc[1])
        texts.append(text)
        
        # Tokenize
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        
        logits = outputs[0]
        pred_label_id = torch.argmax(logits, dim=1).item()
        pred_confidence = torch.softmax(logits, dim=1)[0][pred_label_id].item()
        
        predictions.append(label_map[pred_label_id])
        prediction_ids.append(pred_label_id)
        
        if (index + 1) % 200 == 0:
            print(f"  Processed {index + 1} rows...")

    # Create results dataframe
    results_df = pd.DataFrame({
        'ABSTRACT_ID': df['ABSTRACT_ID'].values[:len(predictions)],
        'TEXT': texts,
        'ORIGINAL_LABEL': df['LABEL'].values[:len(predictions)],
        'PREDICTION': predictions,
        'PREDICTION_ID': prediction_ids
    })

    # Save predictions
    output_file = 'test_predictions_with_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to '{output_file}'")

    # Analysis & visualization
    print("\n" + "="*60)
    print("PART C: ANALYZE AND MODIFICATION")
    print("="*60)

    # Prediction distribution
    print("\n1. PREDICTION DISTRIBUTION")
    print("-" * 60)
    pred_counts = results_df['PREDICTION'].value_counts()
    print(pred_counts)
    print(f"\nTotal predictions: {len(results_df)}")
    
    # Percentages
    print("\nPercentage Distribution:")
    for label in label_map.values():
        count = (results_df['PREDICTION'] == label).sum()
        pct = (count / len(results_df)) * 100
        print(f"  {label:25s}: {count:6d} ({pct:5.2f}%)")

    # Generate visualizations
    print("\n2. Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BERT Model - Prediction Analysis on Test Data', fontsize=16, fontweight='bold')

    # Subplot 1: Prediction Distribution Bar Chart
    ax1 = axes[0, 0]
    pred_counts.sort_values(ascending=False).plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Prediction Distribution', fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(pred_counts.sort_values(ascending=False)):
        ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Subplot 2: Percentage Distribution Pie Chart
    ax2 = axes[0, 1]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax2.set_title('Percentage Distribution', fontweight='bold')

    # Subplot 3: Cumulative Distribution
    ax3 = axes[1, 0]
    sorted_counts = pred_counts.sort_values(ascending=False)
    cum_pct = (sorted_counts.cumsum() / len(results_df) * 100).values
    ax3.plot(range(len(sorted_counts)), cum_pct, marker='o', linewidth=2, markersize=8)
    ax3.set_xticks(range(len(sorted_counts)))
    ax3.set_xticklabels(sorted_counts.index, rotation=45)
    ax3.set_ylabel('Cumulative %')
    ax3.set_title('Cumulative Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])

    # Subplot 4: Sample Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Total Samples Tested: {len(results_df)}
    Number of Categories: {len(label_map)}
    
    Model: BERT (bert-base-uncased)
    Training Categories: 5 (Flu, Pregnancy, Sexual Intercourse, Period, Exercise)
    
    Prediction Breakdown:
    """
    
    for label in label_map.values():
        count = (results_df['PREDICTION'] == label).sum()
        pct = (count / len(results_df)) * 100
        summary_text += f"\n    {label:25s}: {count:6d} ({pct:5.1f}%)"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    print("  -> Saved 'confusion_matrix_analysis.png'")
    plt.close()

    # Create prediction distribution heatmap
    print("\n3. Creating Prediction Distribution Heatmap...")
    
    # Create a matrix showing prediction distribution
    # Rows = all categories (for symmetry), Columns = predicted categories
    matrix_data = []
    all_labels = list(label_map.values())
    
    for actual_label in all_labels:
        row = []
        for pred_label in all_labels:
            # Count predictions (without ground truth, we show prediction totals)
            count = (results_df['PREDICTION'] == pred_label).sum()
        row.append(count)
        matrix_data.append(row)
    
    # Create a simpler distribution matrix (label vs prediction count)
    pred_dist = np.zeros((1, len(label_map)))
    for i, label in enumerate(all_labels):
        pred_dist[0, i] = (results_df['PREDICTION'] == label).sum()
    
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(pred_dist.astype(int), annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=all_labels, yticklabels=['Predictions'],
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Prediction Distribution Heatmap', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('prediction_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    print("  -> Saved 'prediction_distribution_heatmap.png'")
    plt.close()

    # Model performance analysis
    print("\n4. MODEL PERFORMANCE ANALYSIS")
    print("-" * 60)
    
    # Most confident categories
    results_df['BINARY_PRED_ID'] = results_df['PREDICTION_ID']
    print("\nPrediction Summary:")
    print(f"  Most common prediction: {pred_counts.idxmax()} ({pred_counts.max()} samples, {pred_counts.max()/len(results_df)*100:.1f}%)")
    print(f"  Least common prediction: {pred_counts.idxmin()} ({pred_counts.min()} samples, {pred_counts.min()/len(results_df)*100:.1f}%)")
    print(f"  Balanced? (std dev of counts): {pred_counts.std():.2f}")

    # Save detailed report
    print("\n5. Saving Detailed Report...")
    report_path = 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BERT TRIAGE MODEL - TEST RESULTS ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total samples tested: {len(results_df)}\n")
        f.write(f"Source: Test Data/test.csv\n")
        f.write(f"Model: BERT (bert-base-uncased)\n\n")
        
        f.write("TRAINING CATEGORIES\n")
        f.write("-"*70 + "\n")
        for id, name in label_map.items():
            f.write(f"  {id}: {name}\n")
        f.write("\n")
        
        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        for label in label_map.values():
            count = (results_df['PREDICTION'] == label).sum()
            pct = (count / len(results_df)) * 100
            f.write(f"  {label:25s}: {count:6d} ({pct:5.2f}%)\n")
        f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*70 + "\n")
        f.write("The model has been trained on 5 medical triage categories and makes\n")
        f.write("predictions on all input text, assigning each to one of these categories.\n")
        f.write("The MeDAL test dataset contains medical abbreviation labels which do not\n")
        f.write("correspond to the training categories, making this a zero-shot prediction\n")
        f.write("scenario where we analyze the model's natural prediction distribution.\n\n")
        
        f.write("The prediction distribution above shows how the model classifies the\n")
        f.write("test samples across the 5 categories. This analysis helps understand:\n")
        f.write("  1. Whether predictions are balanced or skewed\n")
        f.write("  2. Model behavior on out-of-domain data\n")
        f.write("  3. Potential biases in the model's decision boundary\n")

    print(f"  -> Saved detailed report to '{report_path}'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {output_file} (predictions)")
    print(f"  - confusion_matrix_analysis.png (visualizations)")
    print(f"  - prediction_distribution_heatmap.png (distribution)")
    print(f"  - {report_path} (detailed analysis)")

if __name__ == "__main__":
    predict_with_analysis()
