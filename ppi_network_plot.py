import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_ppi(ppi_edges_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(ppi_edges_csv)

    G = nx.from_pandas_edgelist(df, 'Source', 'Target')
    plt.figure(figsize=(8,6))
    nx.draw(G, with_labels=True, node_color='#2ca02c', node_size=500, font_size=10)
    plt.tight_layout()
    plt.savefig(output_dir/'PPI_network.svg', bbox_inches='tight')
    plt.savefig(output_dir/'PPI_network.png', dpi=300, bbox_inches='tight')
    return plt.gcf()
