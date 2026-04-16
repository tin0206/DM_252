import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Create a folder to store graphs
output_dir = 'eda_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Load data
df = pd.read_csv('train.csv')

# --- Graph 1: Label Distribution ---
plt.figure(figsize=(10, 6))
sns.countplot(x='Label', data=df, palette='viridis')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig(f'{output_dir}/label_distribution.png', dpi=300) # Save graph
plt.close() # Close to free up memory

# --- Graph 2: Text Length Distribution ---
df['title_len'] = df['title'].apply(lambda x: len(str(x).split()))
df['abstract_len'] = df['abstract'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['title_len'], kde=True, color='blue')
plt.title('Title Length Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['abstract_len'], kde=True, color='green')
plt.title('Abstract Length Distribution')
plt.savefig(f'{output_dir}/text_length_distribution.png', dpi=300)
plt.close()

# --- Graph 3: Top Venues ---
plt.figure(figsize=(12, 6))
df['venue'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Venues')
plt.xticks(rotation=45)
plt.tight_layout() # Adjust layout to not cut off labels
plt.savefig(f'{output_dir}/top_venues.png', dpi=300)
plt.close()

# --- Graph 4: Publication Trend ---
plt.figure(figsize=(12, 6))
df.groupby('year')['id'].count().plot(kind='line', marker='o', color='red')
plt.title('Publication Trend Over Years')
plt.grid(True)
plt.savefig(f'{output_dir}/publication_trend.png', dpi=300)
plt.close()

print(f"Success: All graphs have been saved in the '{output_dir}' folder.")