import pandas as pd

def string_ppi_expand(targets_csv, output_csv):
    df = pd.read_csv(targets_csv)
    # Dummy PPI edges
    edges = pd.DataFrame({'Source': ['AKT1'], 'Target': ['TNF']})
    edges.to_csv(output_csv, index=False)
    return edges
