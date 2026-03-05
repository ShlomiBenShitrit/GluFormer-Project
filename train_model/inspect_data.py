import torch
import sys
import pandas.core.indexes.base
import numpy as np

# גשר תאימות (חובה)
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

def inspect_full(file_name):
    print(f"\n{'#'*60}")
    print(f" FILE: {file_name}")
    print(f"{'#'*60}")
    
    # טעינת הקובץ
    data = torch.load(file_name, weights_only=False)
    tokens = data['tokens'] 
    
    print(f"📊 Total Samples: {len(tokens)}")
    
    # הגדרת תצוגה של numpy כדי שלא יקצר את המערכים (עם ...)
    np.set_printoptions(threshold=sys.maxsize)

    for i in range(len(tokens)):
        sample = tokens[i]
        # חילוץ הערכים הגולמיים
        values = sample.values if hasattr(sample, 'values') else sample
        
        # הדפסה של כל הדגימה
        print(f"\n--- SAMPLE {i} ---")
        print(list(values)) # מדפיס כרשימה פייתונית מלאה
        
    print(f"\n{'='*60}")
    print(f"✅ Finished printing all {len(tokens)} samples.")

if __name__ == "__main__":
    # נריץ קודם על ה-Train ואז על ה-Val
    inspect_full('cgm_CGMacros_diet_filtered_processed_aligned_tokenized_tensors_train_fixed.pt')
    inspect_full('cgm_CGMacros_diet_filtered_processed_aligned_tokenized_tensors_val_fixed.pt')