import torch
import sys
import pandas as pd
import pandas.core.indexes.base
import numpy as np

# Compatibility bridge for legacy pandas objects
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

def inspect_full(file_name):
    print(f"\n{'#'*60}")
    print(f" FILE: {file_name}")
    print(f"{'#'*60}")
    
    try:
        # Load the file
        data = torch.load(file_name, weights_only=False)
        
        # Handle cases where data might be a dict or a direct list
        tokens = data['tokens'] if isinstance(data, dict) and 'tokens' in data else data
        
        print(f"📊 Total Samples: {len(tokens)}")
        
        # Set numpy options to print full arrays
        np.set_printoptions(threshold=sys.maxsize)

        for i in range(len(tokens)):
            sample = tokens[i]
            
            # Safe value extraction for both Tensors and Pandas objects
            if torch.is_tensor(sample):
                # Using .tolist() is the safest way to print a full PyTorch tensor
                values = sample.detach().cpu().numpy().tolist()
            elif hasattr(sample, 'values') and not callable(sample.values):
                # For Pandas Series
                values = sample.values.tolist()
            elif hasattr(sample, 'tolist'):
                values = sample.tolist()
            else:
                values = list(sample)
            
            # Print the sample
            print(f"\n--- SAMPLE {i} ---")
            print(values)
            print(f"Length: {len(values)}")
            
        print(f"\n{'='*60}")
        print(f"✅ Finished printing all {len(tokens)} samples.")

    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

if __name__ == "__main__":
    # Check both training and validation fixed files
    inspect_full('cgm_diet_filtered_processed_aligned_tokenized_tensors_val.pt')