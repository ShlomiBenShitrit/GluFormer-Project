# import sys
# import pandas
# import pandas.core.indexes.base
# import torch
# import pickle
# import os

# # 1. Compatibility Bridge (Monkey Patch)
# # Redirects legacy pandas internal paths to modern ones
# sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

# # Support for Int64/Float64 indexes that were merged into the base Index in Pandas 2.0
# if not hasattr(pandas.core.indexes.base, 'Int64Index'):
#     pandas.core.indexes.base.Int64Index = pandas.core.indexes.base.Index
# if not hasattr(pandas.core.indexes.base, 'Float64Index'):
#     pandas.core.indexes.base.Float64Index = pandas.core.indexes.base.Index

# def fix_and_save(file_name):
#     if not os.path.exists(file_name):
#         print(f"❌ Error: File {file_name} not found.")
#         return

#     print(f"🔄 Attempting to extract {file_name}...")
    
#     data = None
    
#     # Method A: Try standard PyTorch load
#     try:
#         data = torch.load(file_name, weights_only=False)
#         print("✅ Success: Loaded using torch.load")
#     except Exception as e:
#         print(f"⚠️ Warning: torch.load failed ({e}). Trying with pickle...")
        
#         # Method B: Try standard Pickle load (common for manually saved dicts)
#         try:
#             with open(file_name, 'rb') as f:
#                 data = pickle.load(f)
#             print("✅ Success: Loaded using pickle.load")
#         except Exception as e2:
#             print(f"❌ Final Error: Could not read file. ({e2})")
#             return

#     if data:
#         # Save again in a proper, modern PyTorch format
#         new_file_name = file_name.replace(".pt", "_fixed.pt")
#         torch.save(data, new_file_name)
#         print(f"💾 Saved fixed format to: {new_file_name}\n")

# if __name__ == "__main__":
#     # List of files to process
#     files = [
#         'cgm_diet_filtered_processed_aligned_tokenized_tensors_train.pt',
#         'cgm_diet_filtered_processed_aligned_tokenized_tensors_val.pt'
#     ]
#     for f in files:
#         fix_and_save(f)

import sys
import pandas as pd
import pandas.core.indexes.base
import torch
import pickle
import os

# 1. Compatibility Bridge (Monkey Patch)
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
if not hasattr(pandas.core.indexes.base, 'Int64Index'):
    pandas.core.indexes.base.Int64Index = pandas.core.indexes.base.Index
if not hasattr(pandas.core.indexes.base, 'Float64Index'):
    pandas.core.indexes.base.Float64Index = pandas.core.indexes.base.Index

def fix_and_save(file_name):
    if not os.path.exists(file_name):
        print(f"❌ Error: File {file_name} not found.")
        return

    print(f"🔄 Processing {file_name}...")
    data = None
    
    # Load data (Try torch then pickle)
    try:
        data = torch.load(file_name, weights_only=False)
        print("✅ Loaded using torch.load")
    except Exception:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        print("✅ Loaded using pickle.load")

    if data and 'tokens' in data:
        # 1. Save fixed .pt version
        new_pt_name = file_name.replace(".pt", "_fixed.pt")
        torch.save(data, new_pt_name)
        print(f"💾 Saved fixed .pt to: {new_pt_name}")

        # 2. Export to CSV for visual inspection
        print(f"📊 Exporting to CSV for inspection...")
        
        # Convert list of Series/Tensors to a DataFrame
        # Each element in data['tokens'] becomes a row in the CSV
        token_list = []
        for item in data['tokens']:
            if hasattr(item, 'values'): # If it's a Pandas Series
                token_list.append(item.values)
            else: # If it's already a tensor or list
                token_list.append(item)
        
        df = pd.DataFrame(token_list)
        
        csv_name = file_name.replace(".pt", "_preview.csv")
        # Save only the first 100 columns if the sequence is too long, 
        # or remove .head(100) to save everything
        df.to_csv(csv_name, index=False)
        print(f"📄 CSV created: {csv_name} (Shape: {df.shape})\n")

if __name__ == "__main__":
    files = [
        'cgm_CGMacros_diet_filtered_processed_aligned_tokenized_tensors_train.pt',
        'cgm_CGMacros_diet_filtered_processed_aligned_tokenized_tensors_val.pt'
    ]
    for f in files:
        fix_and_save(f)