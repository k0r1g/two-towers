import pandas as pd

# Read the TSV file
df = pd.read_csv('pairs.tsv', sep='\t', header=None, names=['query', 'document', 'label'])

# Convert label to integer
df['label'] = df['label'].astype(int)

# Save as parquet
df.to_parquet('pairs.parquet', index=False)

print("Converted pairs.tsv to pairs.parquet") 