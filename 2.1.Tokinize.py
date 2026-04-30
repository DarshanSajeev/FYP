import json
import torch
from transformers import AutoTokenizer

def main():
    # Load ChatDoctor tokenizer
    print("Loading ChatDoctor Tokenizer")
    # Using the official ChatDoctor repository
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)

    # Ensure tokenizer has a pad token required for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    input_filename = "../Training Data/triage_berts_dataset.json"
    print(f"Loading data from {input_filename}...")
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_filename}")
        return

    print(f"Loaded {len(dataset)} items.")

    # Tokenize dataset
    print("Tokenizing...")
    input_ids = []
    attention_masks = []
    labels = []

    for item in dataset:
        raw_text = item['text']
        label = item['label']

        # Split text into question and answer (if present)
        try:
            parts = raw_text.split(" [SEP] ")
            question = parts[0]
            answer = parts[1] if len(parts) > 1 else ""
        except:
            question = raw_text
            answer = ""
        # Concatenate into a conversational prompt for LLaMA
        if answer:
            combined_text = f"Patient: {question}\nChatDoctor: {answer}"
        else:
            combined_text = f"Patient: {question}"

        # Tokenize the combined string
        encoded_dict = tokenizer(
            combined_text,
            max_length=128,  # You can increase this to 256 if your patient queries are long
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

    print("\nFinished")
    print(f"Tokenized {len(labels)} items.")
    
    # Save tokenized tensors to file
    output_filename = '../ChatDoctor/chatdoctor_triage_tensors.pt'
    import os
    os.makedirs('../ChatDoctor', exist_ok=True)
    torch.save({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }, output_filename)
    print(f"Saved output to '{output_filename}'")

if __name__ == "__main__":
    main()