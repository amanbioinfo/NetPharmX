import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_enrichment(targets_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(targets_csv)
    enrichment = df['Target'].value_counts().reset_index()
    enrichment.columns = ['Term', 'Count']

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'figure.figsize': (8,6)
    })
    sns.set_palette(["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])

    fig, ax = plt.subplots()
    sns.barplot(data=enrichment, x='Count', y='Term', ax=ax)
    ax.set_title('Target Enrichment')
    ax.set_xlabel('Count')
    ax.set_ylabel('Target')
    plt.tight_layout()
    
    fig.savefig(output_dir/'enrichment_barplot.svg', bbox_inches='tight')
    fig.savefig(output_dir/'enrichment_barplot.png', dpi=300, bbox_inches='tight')
    return fig
