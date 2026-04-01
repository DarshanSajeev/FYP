import json
import torch
from transformers import AutoTokenizer

def main():
    # 1. Download Hugging Face version of the ChatDoctor tokenizer
    print("Loading ChatDoctor Tokenizer")
    # Using the official ChatDoctor repository
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    
    # 🔥 CRITICAL FIX: LLaMA tokenizers don't have a default padding token, 
    # which PyTorch absolutely needs in order to stack tensors into batches.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    input_filename = "../Training Data/triage_berts_dataset.json"
    print(f"Loading data from {input_filename}...")
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_filename}")
        return

    print(f"Loaded {len(dataset)} items.")

    # 3. Start tokenizing
    print("Tokenizing...")
    input_ids = []
    attention_masks = []
    labels = []

    for item in dataset:
        raw_text = item['text']
        label = item['label']

        # Split the text back into Question and Answer (from your original prep)
        try:
            parts = raw_text.split(" [SEP] ")
            question = parts[0]
            answer = parts[1] if len(parts) > 1 else ""
        except:
            question = raw_text
            answer = ""

        # 🔥 LLaMA does not use [SEP] tokens to separate sentences like BERT.
        # We must manually concatenate the text into a single conversational prompt.
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

    # 4. Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print("\nFinished")
    print(f"Tokenized {len(labels)} items.")
    
    # 5. Save the tokenized data as a PyTorch file for use in QLoRA training
    output_filename = '../ChatDoctor/chatdoctor_triage_tensors.pt'
    
    # Ensure you have the ChatDoctor folder created before saving
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