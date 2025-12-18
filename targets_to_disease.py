import pandas as pd

def map_targets_to_disease(targets_csv, output_csv):
    df = pd.read_csv(targets_csv)
    # Dummy mapping; replace with actual target→disease mapping
    df['Disease'] = df['Target'].map({'AKT1': 'Cancer', 'TNF': 'Inflammation'})
    df.to_csv(output_csv, index=False)
    return df
