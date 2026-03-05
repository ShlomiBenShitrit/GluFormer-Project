import torch
import random
import sys
import numpy as np
import pandas.core.indexes.base

# Compatibility bridge for legacy pandas objects
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

def split_and_prepare_data(source_file, train_ratio=0.8):
    print(f"--- Data Split Utility ---")
    print(f"Loading source file: {source_file}")
    
    # Load the data
    data = torch.load(source_file, weights_only=False)
    
    # Extract tokens list
    all_samples = data['tokens'] if isinstance(data, dict) and 'tokens' in data else data
    total_samples = len(all_samples)
    print(f"Total unique subjects found: {total_samples}")

    # Process samples into a consistent tensor format to avoid layout/sparse errors
    processed_samples = []
    for s in all_samples:
        if torch.is_tensor(s):
            processed_samples.append(s.detach().cpu())
        elif hasattr(s, 'to_numpy'): # Handle Pandas Series
            processed_samples.append(torch.from_numpy(s.to_numpy()))
        else:
            processed_samples.append(torch.tensor(s))

    # Shuffle subjects to ensure random distribution
    random.seed(42)
    random.shuffle(processed_samples)
    
    # Calculate split point
    split_idx = int(total_samples * train_ratio)
    
    train_data = processed_samples[:split_idx]
    val_data = processed_samples[split_idx:]
    
    # Save processed splits
    train_out = 'train_cgmacros_split.pt'
    val_out = 'val_cgmacros_split.pt'
    
    torch.save({'tokens': train_data}, train_out)
    torch.save({'tokens': val_data}, val_out)
    
    print(f"\nSuccess: Data split completed.")
    print(f"Training set: {len(train_data)} subjects saved to {train_out}")
    print(f"Validation set: {len(val_data)} subjects saved to {val_out}")
    print(f"--------------------------")

if __name__ == "__main__":
    # Use the 'fixed' file generated from your CGMacros processing
    target_file = 'cgm_CGMacros_diet_filtered_processed_aligned_tokenized_tensors_train.pt'
    split_and_prepare_data(target_file)