import os
import glob
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
import io

def manual_smart_parse(base_path, output_cgm, output_sqlog):
    print(f"Searching for PhysioNet data starting from: {base_path}")

    # Recursively locate all CSV files matching the participant data pattern
    csv_files = glob.glob(os.path.join(base_path, "**", "CGMacros-*.csv"), recursive=True)
    if not csv_files:
        print("No files found. Please check the path.")
        return

    print(f"Found {len(csv_files)} participant files. Starting smart manual conversion...")

    # Define headers for the two output streams: continuous glucose (CGM) and dietary logs (sqlog)
    cgm_rows = ["RegistrationCode,Date,GlucoseValue,PPGR\n"]
    sqlog_rows = ["RegistrationCode,Date,energy_kcal,carbohydrate_g,protein_g,totallipid_g,sugarstotal_g,caffeine_mg,water_g,alcohol_g,cholesterol_mg,meal_type,score\n"]

    # Set a reference base date to synchronize relative timestamps into absolute datetime format
    base_date = datetime(2021, 1, 1, 0, 0, 0)

    for file_path in tqdm(csv_files, desc="Parsing files"):
        # Extract participant ID from filename and format it as a registration code (e.g., 10Kxxx)
        file_name = os.path.basename(file_path)
        num_id = file_name.lower().replace('cgmacros-', '').replace('.csv', '')
        reg_code = f"10K{num_id}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read and normalize headers to identify data columns across different device formats
            header_line = f.readline().strip().lower()
            headers = [h.strip() for h in header_line.split(',')]

            # Dynamic mapping: identify column indices for glucose sensors and nutritional metrics
            idx_map = {}
            for i, h in enumerate(headers):
                if 'libre' in h: idx_map['libre'] = i
                elif 'dexcom' in h: idx_map['dexcom'] = i
                elif 'meal type' in h or 'meal' in h: idx_map['meal_type'] = i
                elif 'calor' in h and 'activity' not in h: idx_map['calories'] = i
                elif 'carb' in h: idx_map['carbs'] = i
                elif 'protein' in h: idx_map['protein'] = i
                elif 'fat' in h or 'lipid' in h: idx_map['fat'] = i
                elif 'timestamp' in h or 'time' in h or 'date' in h:
                    if 'timestamp' not in idx_map: idx_map['timestamp'] = i
                elif 'image' in h or 'photo' in h: idx_map['image'] = i

            # Skip files that lack essential temporal information
            if 'timestamp' not in idx_map:
                continue
                
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')

                # Helper function to safely extract values based on the dynamic index map
                def get_val(key):
                    if key in idx_map and idx_map[key] < len(parts):
                        return parts[idx_map[key]].strip()
                    return ""
                    
                raw_timestamp = get_val('timestamp')
                libre_gl = get_val('libre')
                dexcom_gl = get_val('dexcom')
                meal_type = get_val('meal_type')
                calories = get_val('calories')
                carbs = get_val('carbs')
                protein = get_val('protein')
                fat = get_val('fat')
                image_path = get_val('image')
                
                if not raw_timestamp:
                    continue

                # Temporal Synchronization: Convert relative minutes or raw strings to standardized timestamps
                if raw_timestamp.replace('.', '', 1).isdigit() and len(raw_timestamp) < 10:
                    try:
                        minutes = int(float(raw_timestamp))
                        timestamp = (base_date + timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        timestamp = raw_timestamp
                else:
                    timestamp = raw_timestamp

                # Multimodal Extraction - Step 1: Consolidate glucose readings from available sensors
                glucose = libre_gl if libre_gl else dexcom_gl
                if glucose:
                    cgm_rows.append(f"{reg_code},{timestamp},{glucose},0.0\n")

                # Validate presence of dietary intake via caloric data or meal images
                has_cal = calories and calories.replace('.', '', 1).isdigit() and float(calories) > 0
                has_img = bool(image_path)

                # Multimodal Extraction - Step 2: Extract macronutrients to create dietary tokens for the model
                if has_cal or has_img:
                    cal_val = calories if (calories and calories.replace('.', '', 1).isdigit()) else "0.0"
                    carb_val = carbs if (carbs and carbs.replace('.', '', 1).isdigit()) else "0.0"
                    prot_val = protein if (protein and protein.replace('.', '', 1).isdigit()) else "0.0"
                    fat_val = fat if (fat and fat.replace('.', '', 1).isdigit()) else "0.0"
                    m_type = meal_type if meal_type else "unknown"

                    # Store integrated nutritional data to be used as multimodal context alongside CGM tokens
                    sqlog_rows.append(f"{reg_code},{timestamp},{cal_val},{carb_val},{prot_val},{fat_val},0.0,0.0,0.0,0.0,0.0,{m_type},0.0\n")

    print("\nResampling CGM data to 15-minute averages...")
    # Load collected CGM data into a Pandas DataFrame
    cgm_df = pd.read_csv(io.StringIO("".join(cgm_rows)))
    # Convert Date column to datetime objects and handle parsing errors
    cgm_df['Date'] = pd.to_datetime(cgm_df['Date'], errors='coerce')
    # Remove rows with missing timestamps or glucose values
    cgm_df = cgm_df.dropna(subset=['Date', 'GlucoseValue'])
    
    # Grouping by participant and calculating average sugar every 15 minutes
    cgm_resampled = (
        cgm_df.groupby('RegistrationCode')
        .apply(lambda x: x.set_index('Date').resample('15T')['GlucoseValue'].mean())
        .reset_index()
    )
    
    # Remove empty rows resulting from resampling and initialize PPGR column
    cgm_resampled = cgm_resampled.dropna(subset=['GlucoseValue'])
    cgm_resampled['PPGR'] = 0.0
    # Format Date column as string for consistency
    cgm_resampled['Date'] = cgm_resampled['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Arranging the columns in the required order
    cgm_resampled = cgm_resampled[['RegistrationCode', 'Date', 'GlucoseValue', 'PPGR']]

    print("Saving CGM data...")
    cgm_resampled.to_csv(output_cgm, index=False)
        
    print("Saving SQLog data...")
    with open(output_sqlog, 'w', encoding='utf-8') as f:
        f.writelines(sqlog_rows)
        
    print(f"--> Success! Extracted {len(sqlog_rows) - 1} meal logs and {len(cgm_resampled)} resampled 15-min CGM readings.")

if __name__ == "__main__":
    INPUT_PATH = "./physionet_data/"
    OUTPUT_CGM = "./CGM_train.csv"
    OUTPUT_SQLOG = "./SQLog_train.csv"
    manual_smart_parse(INPUT_PATH, OUTPUT_CGM, OUTPUT_SQLOG)
