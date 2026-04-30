import torch
import pandas as pd
import os
from transformers import BertTokenizer, BertForSequenceClassification

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label map for 20 categories
label_map = {
    0: "Respiration",
    1: "Gastroenterology",
    2: "Cardiology",
    3: "Neurology",
    4: "Dermatology",
    5: "Orthopedics",
    6: "Oncology",
    7: "Endocrinology",
    8: "Nephrology",
    9: "Hepatology",
    10: "Psychiatry",
    11: "Ophthalmology",
    12: "Otolaryngology",
    13: "Urology",
    14: "Gynecology",
    15: "Pediatrics",
    16: "Rheumatology",
    17: "Immunology",
    18: "Infectious Diseases",
    19: "Other"
}

def predict_medal_test_data():
    # Predict labels for test.csv and save results to 'predicted_results_medal.csv'
    print(f"Loading model on {device}...")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=20,
        output_attentions=False,
        output_hidden_states=False
    )

    state_dict = torch.load(
        "bert_model_fold_0.pth",
        map_location=device,
        weights_only=True   # removes warning
    )

    model.load_state_dict(state_dict, strict=False)

    print("Weights loaded successfully.")

    model.to(device)
    model.eval()


    # Load test CSV
    csv_path = os.path.join("../Test Data", "test.csv") 
    print(f"Reading data from '{csv_path}'...")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'.")
        print("Make sure 'test.csv' is inside the 'Test Data' folder.")
        return
        
    # Read CSV (limit 2000 rows)
    try:
        # We read 2000 rows for the report statistics
        df = pd.read_csv(csv_path, nrows=2000)
        print(f"Successfully loaded {len(df)} rows for testing.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    predictions = []

    print("Running predictions (this may take 1-2 minutes)...")
    
    for index, row in df.iterrows():
        # Handle different column names
        if 'text' in row:
            text = str(row['text'])
        else:
            text = str(row.iloc[0])
        
        # Tokenize text
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',  # Updated from pad_to_max_length
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
        
        predictions.append({
            'id': index,
            'text_snippet': text[:60] + "...",
            'predicted_label_id': pred_label_id,
            'prediction_name': label_map.get(pred_label_id, "Unknown")
        })
        
        if index % 200 == 0:
            print(f"Processed {index} rows...")

    # Save results to CSV
    output_file = 'predicted_results_medal.csv'
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("FINAL PREDICTIONS ON MeDAL DATA")
    print("="*50)
    
    # Calculate percentages for the report
    counts = results_df['prediction_name'].value_counts()
    print("\nPrediction Distribution:")
    print(counts)
    
    print(f"\nSaved full results to '{output_file}'")

if __name__ == "__main__":
    predict_medal_test_data()