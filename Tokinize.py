import json
import torch
from transformers import BertTokenizer

def main():
    # Download PyTorch version of BERT tokenizer
    print("Loading BERT Tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # load data
    input_filename = "triage_berts_dataset.json"
    print(f"Loading data from {input_filename}...")
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find triage_berts_dataset.json")
        return

    print(f"Loaded {len(dataset)} items.")

    # start tokenizing
    print("Tokenizing data...")
    input_ids = []
    attention_masks = []
    labels = []

    for item in dataset:
        raw_text = item['text']
        label = item['label']

        # Split the text back into Question and Answer
        try:
            parts = raw_text.split(" [SEP] ")
            question = parts[0]
            answer = parts[1] if len(parts) > 1 else ""
        except:
            question = raw_text
            answer = ""

        # Tokenize the pair (question, answer)
        encoded_dict = tokenizer(
            # Automatically add [CLS] and [SEP]
            question,
            answer,
            add_special_tokens=True,
            # Adds a max length of patient's response
            max_length=128,
            # Pads short responses to max_length
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(torch.tensor(label))

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print("\n--- SUCCESS ---")
    print(f"Tokenized {len(labels)} items.")
    
    torch.save({
        'input_ids': input_ids, 
        'attention_mask': attention_masks, 
        'labels': labels
    }, 'triage_tensors.pt')
    
    print("Saved output to 'triage_tensors.pt'")

if __name__ == "__main__":
    main()