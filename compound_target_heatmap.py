import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_heatmap(targets_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(targets_csv)
    heatmap_data = pd.crosstab(df['Compound'], df['Target'])

    plt.rcParams.update({'font.size':12,'font.family':'sans-serif'})
    sns.set_palette(["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(heatmap_data, cmap='viridis', ax=ax)
    ax.set_title('Compound–Target Heatmap')
    plt.tight_layout()
    fig.savefig(output_dir/'compound_target_heatmap.svg', bbox_inches='tight')
    fig.savefig(output_dir/'compound_target_heatmap.png', dpi=300, bbox_inches='tight')
    return fig
