import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_target_pathway_heatmap(enrichment_txt, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(enrichment_txt, sep='\t')
    heatmap_data = df.pivot_table(index='Term', columns='Gene', values='Adjusted P-value', fill_value=0)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(heatmap_data, cmap='magma', ax=ax)
    ax.set_title('Target–Pathway Heatmap')
    plt.tight_layout()
    fig.savefig(output_dir/'target_pathway_heatmap.svg', bbox_inches='tight')
    fig.savefig(output_dir/'target_pathway_heatmap.png', dpi=300, bbox_inches='tight')
    return fig
