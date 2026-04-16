import pandas as pd

try:
    df = pd.read_csv('Stage_1_publcitrain.csv')
    
    # 1. Tổng số dòng (tương ứng với số lượng bản ghi)
    total_rows = len(df)
    
    # 2. Số lượng ID duy nhất (đề phòng có ID bị lặp lại)
    unique_ids = df['id'].nunique()
    
    print(f"Total number of entries (rows): {total_rows}")
    print(f"Number of unique IDs: {unique_ids}")

    # Kiểm tra xem có ID nào bị lặp không
    if total_rows != unique_ids:
        print(f"Warning: Found {total_rows - unique_ids} duplicated IDs!")
    else:
        print("Success: Every row has a unique ID.")

except FileNotFoundError:
    print("Error: File 'Stage_1_publcitrain.csv' not found.")