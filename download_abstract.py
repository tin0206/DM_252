import pandas as pd
import requests
import time

def get_abstract(doi):
    """
    Fetch abstract from Semantic Scholar API using DOI.
    """
    if pd.isna(doi) or doi == "":
        return None
    
    # API URL
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
    
    try:
        # Semantic Scholar free tier limit is roughly 1 request/second without API key
        time.sleep(1.1)
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('abstract')
        elif response.status_code == 429:
            print(f"Rate limit hit for DOI {doi}. Sleeping...")
            time.sleep(5) # Wait longer if rate limited
            return None
        else:
            return None
    except Exception as e:
        print(f"Error fetching DOI {doi}: {e}")
        return None

def process_data(input_file, output_file):
    # 1. Load the Stage 1 file
    print("Reading input file...")
    df_stage1 = pd.read_csv(input_file)
    
    # 2. Fetch abstracts
    print("Fetching abstracts from Semantic Scholar (this may take a while)...")
    # We only call API for rows where DOI exists
    df_stage1['abstract'] = df_stage1['doi'].apply(get_abstract)
    
    # 3. Select and rename columns to match the required format
    # Based on your image: id, title, abstract, authors, venue, year
    # We ensure these columns exist in the final dataframe
    column_mapping = {
        'id': 'id',
        'title': 'title',
        'abstract': 'abstract',
        'authors': 'authors',
        'venue': 'venue',
        'year': 'year',
    }
    
    # Create the new dataframe with desired columns
    train_df = df_stage1[list(column_mapping.keys())].copy()
    
    # 4. Sort by ID as requested
    train_df = train_df.sort_values(by='id').reset_index(drop=True)
    
    # 5. Save to train.csv
    train_df.to_csv(output_file, index=False)
    print(f"Success! File saved as {output_file}")

# Execute
if __name__ == "__main__":
    process_data('test (2).csv', 'test.csv')