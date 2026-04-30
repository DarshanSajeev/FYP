import json
import os
from collections import Counter
from tabulate import tabulate  # <--- NEW IMPORT

def main():
    # Load the 4 files into 1 list
    files_to_load = [
        "../Training Data/ehealthforumQAs.json",
        "../Training Data/icliniqQAs.json",
        "../Training Data/questionDoctorQAs.json",
        "../Training Data/webmdQAs.json"
    ]
    
    # Output file for the processed dataset
    output_filename = "../Training Data/triage_berts_dataset.json"

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

    # Step 1: Collect and count all tags
    print("\n--- ANALYZING TAG FREQUENCIES ---")
    tag_counts = Counter()
    
    for item in all_data:
        raw_tag = item.get('tags')
        
        # Handle case where tags might be a list or a single string
        if isinstance(raw_tag, list) and len(raw_tag) > 0:
            tag = raw_tag[0]
        else:
            tag = raw_tag
        
        if tag:
            tag_counts[tag] += 1
    
    print(f"Total unique tags found: {len(tag_counts)}")
    
    # Step 2: Select the top 20 most frequent tags
    top_20_tags = tag_counts.most_common(20)
    indexed_tags = {i: tag for i, (tag, count) in enumerate(top_20_tags)}
    tag_to_index = {tag: i for i, tag in indexed_tags.items()}
    
    print(f"\nTop 20 most frequent tags:")
    for idx, (tag, count) in enumerate(top_20_tags):
        print(f"  {idx}: {tag} ({count} occurrences)")
    
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
        
        # Ensure current_tag is a string (not a list)
        if isinstance(current_tag, list):
            if len(current_tag) > 0:
                current_tag = current_tag[0]
            else:
                continue

        # Only include items with tags in the top 20
        if current_tag in tag_to_index:
            # Get Fields
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # Skip empty items
            if not question or not answer:
                continue

            # Format: Question + [SEP] + Answer
            combined_text = f"{question} [SEP] {answer}"
            
            # Get Label ID from the top 20 tags
            label_id = tag_to_index[current_tag]
            
            # Add to final list
            entry = {
                "text": combined_text,
                "label": label_id,
                "original_tag": current_tag
            }
            processed_dataset.append(entry)
    
    # Count the tags in our processed final list
    final_counts = Counter([item['original_tag'] for item in processed_dataset])
    
    # Print the summary table
    table_rows = []
    
    # Iterate through top 20 tags to keep the printed order consistent
    for idx, (tag, count) in enumerate(top_20_tags):
        final_count = final_counts[tag]
        table_rows.append([idx, tag, final_count])
        
    # Add a final TOTAL row
    table_rows.append(["", "TOTAL", len(processed_dataset)])

    print("\n" + tabulate(table_rows, headers=["ID", "Tag", "Count"], tablefmt="grid"))
    print("\n")

    
    print(f"--- SAVING TO '{output_filename}' ---")
    print(f"Final Dataset Size: {len(processed_dataset)} items")
    
    # Save to triage_berts_dataset.json
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, indent=4)
    
    print(f"Successfully saved to '{output_filename}'")

if __name__ == "__main__":
    main()