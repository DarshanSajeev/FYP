import json
import torch
from transformers import BertTokenizer


def main():

    # Download PyTorch version of BERT tokenizer

    print("Loading BERT Tokenizer")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    # load dataset

    input_filename = "../Training Data/triage_berts_dataset.json"

    print(f"Loading data from {input_filename}...")

   

    try:

        with open(input_filename, 'r', encoding='utf-8') as f:

            dataset = json.load(f)

    except FileNotFoundError:

        print("Error: Could not find Training Data/triage_berts_dataset.json")

        return


    print(f"Loaded {len(dataset)} items.")


    # start tokenizing

    print("Tokenizing")

    input_ids = []

    attention_masks = []

    labels = []


    #Each item looks like this: {"text": "Question [SEP] Answer", "label": 0-4, "original_tag":"tag"}

    for item in dataset:

        raw_text = item['text']

        label = item['label']


        # Split the text back into Question and Answer

        try:

            parts = raw_text.split(" [SEP] ")

            question = parts[0]

            if len(parts) > 1:

                answer = parts[1]

            else:

                answer = ""

        except:

            question = raw_text

            answer = ""


        # Tokenize the question and answer together

        # get input_ids and attention_mask for each item

        #https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PythonBackend

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

            # Masks tells the model which tokens are padding and which are not

            return_attention_mask=True,

            # Returns PyTorch tensors

            return_tensors='pt'

        )


        # Puts the input_ids, attention_masks and labels into lists

        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

        labels.append(torch.tensor(label))


    # Convert lists to tensors

    input_ids = torch.cat(input_ids, dim=0)

    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(labels)


    print("\nFinised")

    print(f"Tokenized {len(labels)} items.")

   

    #Saves the tokenized data as a PyTorch file for use in training BERT

    torch.save({

        'input_ids': input_ids,

        'attention_mask': attention_masks,

        'labels': labels

    }, 'triage_tensors.pt')

   

    print("Saved output to 'Training Data/triage_tensors.pt'")


if __name__ == "__main__":

    main() 