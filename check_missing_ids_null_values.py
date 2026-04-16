import pandas as pd

# --- PART 1: CHECK FOR MISSING IDs IN INPUT FILE ---
print("=== CHECKING FOR MISSING IDs IN STAGE_1_PUBLCITRAIN.CSV ===")
try:
    df_input = pd.read_csv('Stage_1_publcitrain.csv')

    # 1. Get unique existing IDs and sort them
    existing_ids = sorted(df_input['id'].unique())

    # 2. Define the ID range (Min to Max)
    min_id = int(min(existing_ids))
    max_id = int(max(existing_ids))

    # 3. Create a set of all IDs that should exist in a continuous range
    full_range_ids = set(range(min_id, max_id + 1))

    # 4. Identify the gaps
    missing_ids = sorted(list(full_range_ids - set(existing_ids)))

    # 5. Output Results
    print(f"Total rows found: {len(df_input)}")
    print(f"Minimum ID: {min_id}")
    print(f"Maximum ID: {max_id}")
    print(f"Number of missing IDs: {len(missing_ids)}")

    if len(missing_ids) > 0:
        # --- NEW: SAVE MISSING IDs TO FILE ---
        with open('missing_ids.txt', 'w') as f:
            for m_id in missing_ids:
                f.write(f"{m_id}\n")
        print("Success: All missing IDs have been saved to 'missing_ids.txt'")
    else:
        print("Success: No IDs are missing between Min and Max.")

except FileNotFoundError:
    print("Error: 'Stage_1_publcitrain.csv' not found.")

print("\n" + "="*50 + "\n")

# --- PART 2: CHECK FOR NULL VALUES IN PROCESSED TRAIN.CSV ---
print("=== CHECKING FOR NULL VALUES IN TRAIN.CSV ===")
try:
    df_train = pd.read_csv('train.csv')

    # 1. Statistical summary of nulls per column
    print("--- Null Value Statistics per Column ---")
    print(df_train.isnull().sum())
    print("-" * 40)

    # 2. Filter rows that contain at least one null value
    null_rows = df_train[df_train.isnull().any(axis=1)]

    if not null_rows.empty:
        print(f"Found {len(null_rows)} rows containing null values.")
        
        # Optional: Save the null report for manual cleaning
        null_rows.to_csv('null_values_report.csv', index=False)
    else:
        print("Success: No null values found in 'train.csv'.")

except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the generation script has finished running.")