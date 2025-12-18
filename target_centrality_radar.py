import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_target_centrality(ppi_edges_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(ppi_edges_csv)
    centrality = df['Source'].value_counts().reset_index()
    centrality.columns = ['Target', 'Degree']

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(centrality['Target'], centrality['Degree'], color='#2ca02c')
    ax.set_title('Target Centrality (Degree)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(output_dir/'target_degree_barplot.svg', bbox_inches='tight')
    fig.savefig(output_dir/'target_degree_barplot.png', dpi=300, bbox_inches='tight')
    return fig
