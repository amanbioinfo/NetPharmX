import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_hub_genes(ppi_edges_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(ppi_edges_csv)
    hubs = df['Source'].value_counts().reset_index()
    hubs.columns = ['Gene', 'Degree']

    plt.rcParams.update({'font.size':12,'font.family':'sans-serif','figure.figsize':(6,4)})
    sns.set_palette(["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])

    fig, ax = plt.subplots()
    sns.barplot(data=hubs, x='Gene', y='Degree', ax=ax)
    ax.set_title('Hub Genes')
    plt.tight_layout()
    fig.savefig(output_dir/'hub_gene_barplot.svg', bbox_inches='tight')
    fig.savefig(output_dir/'hub_gene_barplot.png', dpi=300, bbox_inches='tight')
    return fig
