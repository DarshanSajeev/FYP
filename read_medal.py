import os
import pandas as pd

# Path to your MeDAL test dataset
file_path = os.path.join("../Test Data", "test.csv")

if not os.path.exists(file_path):
    print(f"Error: Could not find the file at {file_path}")
else:
    try:
        # Read exactly the first 3 rows
        df = pd.read_csv(file_path, nrows=3)
        
        # Print as a formatted markdown table
        print("==================================================")
        print(" FIRST 3 LINES OF MeDAL TEST DATASET")
        print("==================================================\n")
        
        # to_markdown() creates a clean, copy-pasteable table
        print(df.to_markdown(index=False))
        print("\n==================================================")
        
    except Exception as e:
        print(f"Error reading the file: {e}")