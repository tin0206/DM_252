import pandas as pd
import re
import string

def advanced_clean(text):
    # Ensure text is a string
    text = str(text).lower()
    
    # Remove LaTeX symbols
    text = re.sub(r'\$', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_training_data(input_file, output_file):
    # Load data
    df = pd.read_csv(
        input_file, 
        sep=',', 
        quotechar='"', 
        escapechar='\\',
        on_bad_lines='warn'
    )
    
    print(f"Initial rows: {len(df)}")
    print("Preprocessing started (handling NaNs)...")
    
    # IMPORTANT: Fill NaN values with an empty string before combining
    # This prevents the whole 'full_context' from becoming NaN
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    df['venue'] = df['venue'].fillna('')
    
    # Step 1: Feature Engineering - Combine available text
    df['full_context'] = df['title'] + " " + df['abstract'] + " " + df['venue']
    
    # Step 2: Clean the combined text
    df['cleaned_text'] = df['full_context'].apply(advanced_clean)
    
    # Step 3: Select only the necessary columns
    final_train = df[['id', 'cleaned_text', 'Label']]
    
    # Step 4: Final Check
    print(f"Processed rows: {len(final_train)}")
    print(f"Sample: {final_train['cleaned_text'].iloc[0][:100]}...")
    
    # Save the file
    final_train.to_csv(output_file, index=False)
    print(f"Success! Saved to: {output_file}")

# Run the pipeline
create_training_data('train.csv', 'train_final.csv')