import json

with open(r"C:\Users\darsh_opqric6\OneDrive - University of Leeds\Yr 3\FYP\Training Data\triage_berts_dataset.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Get all unique label ID + category name pairs
labels = {}
for item in dataset:
    label_id = item['label']
    category = item.get('category_name', item.get('original_tag', 'unknown'))
    labels[label_id] = category

# Print all 20 sorted by ID
for k in sorted(labels.keys()):
    print(f"{k}: {labels[k]}")