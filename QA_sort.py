import json
import os
from collections import Counter
from tabulate import tabulate  # <--- NEW IMPORT

def main():
    # Load the 4 files into 1 list
    files_to_load = [
        "ehealthforumQAs.json",
        "icliniqQAs.json",
        "questionDoctorQAs.json",
        "webmdQAs.json"
    ]
    
    # Output file for the processed dataset
    output_filename = "triage_berts_dataset.json"

    # Need to load all the files together
    print("- LOADING DATA -")
    all_data = []
    
    for filename in files_to_load:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    file_content = json.load(f)
                    # Expecting each file to contain a list of Q&A items
                    if isinstance(file_content, list):
                        all_data.extend(file_content)
                        print(f"Loaded {len(file_content)} items from {filename}")
                    else:
                        print(f"Warning: {filename} content is not a list. Skipped.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename} (Skipping for this run)")

    print(f"Total Combined Data Count: {len(all_data)}")

    # Need to sort into the top 5 tags from the paper
    print("\n--- TAGS ---")
    
    # Dictionary to make incrimenting each tag easier
    indexed_tags = {
        "pregnancy": 0,
        "period": 1,
        "sexual intercourse": 2,
        "exercise": 3,
        "flu": 4
    }
    
    # Create the list of tags from the keys of the dictionary
    target_tags = list(indexed_tags.keys())
    top_5_tags = target_tags
    print(f"Indexed Tags: {indexed_tags}")
    
    print("\n--- FILTERING & FORMATTING ---")
    processed_dataset = []

    for item in all_data:
        # Get the tag for the current item
        raw_tag = item.get('tags')
        
        # Handle case where tags might be a list or a single string
        if isinstance(raw_tag, list) and len(raw_tag) > 0:
            current_tag = raw_tag[0]
        else:
            current_tag = raw_tag

        if current_tag in top_5_tags:
            # Get Fields
            question = item.get('question', '')
            answer = item.get('answer', '')

            # Format: Question + [SEP] + Answer
            # This is based off of Figure 2 in the paper
            combined_text = f"{question} [SEP] {answer}"
            
            # Get Label ID
            label_id = indexed_tags[current_tag]

            # Add to final list
            entry = {
                "text": combined_text,
                "label": label_id,
                "original_tag": current_tag
            }
            processed_dataset.append(entry)
    
    # Count the tags in our processed final list
    final_counts = Counter([item['original_tag'] for item in processed_dataset])
    
    # Print the summary table (Updated to use tabulate)
    table_rows = []
    
    # Iterate through target_tags to keep the printed order consistent
    for tag in target_tags:
        label = indexed_tags[tag]
        count = final_counts[tag]
        table_rows.append([tag, label, count])
        
    # Add a final TOTAL row
    table_rows.append(["TOTAL", "", len(processed_dataset)])

    # Refrence I used for table formatting:
    # https://www.geeksforgeeks.org/python/how-to-make-a-table-in-python/
    print("\n" + tabulate(table_rows, headers=["Category", "Label", "Count"], tablefmt="grid"))
    print("\n")

    
    print(f"--- SAVING TO '{output_filename}' ---")
    print(f"Final Dataset Size: {len(processed_dataset)} items")
    
    # Save to triage_berts_dataset.json
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, indent=4)
    
    print(f"Successfully saved to '{output_filename}'")

if __name__ == "__main__":
    main()