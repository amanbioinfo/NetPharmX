import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_network(targets_csv, disease_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df_targets = pd.read_csv(targets_csv)
    df_disease = pd.read_csv(disease_csv)

    G = nx.Graph()
    for _, row in df_targets.iterrows():
        G.add_node(row['Compound'], type='compound')
        G.add_node(row['Target'], type='target')
        G.add_edge(row['Compound'], row['Target'])
    for _, row in df_disease.iterrows():
        G.add_node(row['Disease'], type='disease')
        G.add_edge(row['Target'], row['Disease'])

    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#1f77b4', node_size=500, font_size=10, font_family='sans-serif')
    plt.tight_layout()
    plt.savefig(output_dir/'compound_target_disease_network.svg', bbox_inches='tight')
    plt.savefig(output_dir/'compound_target_disease_network.png', dpi=300, bbox_inches='tight')
    return plt.gcf()
