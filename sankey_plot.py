import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def plot_sankey(targets_csv, disease_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df_targets = pd.read_csv(targets_csv)
    df_disease = pd.read_csv(disease_csv)

    # Prepare nodes and links
    nodes = list(set(df_targets['Compound']).union(df_targets['Target']).union(df_disease['Disease']))
    node_indices = {n:i for i,n in enumerate(nodes)}
    links = []
    for _, r in df_targets.iterrows(): links.append({'source':node_indices[r['Compound']], 'target':node_indices[r['Target']], 'value':1})
    for _, r in df_disease.iterrows(): links.append({'source':node_indices[r['Target']], 'target':node_indices[r['Disease']], 'value':1})

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color="#1f77b4"),
        link=dict(source=[l['source'] for l in links], target=[l['target'] for l in links], value=[l['value'] for l in links])
    ))
    fig.write_html(output_dir/'sankey_compound_target_disease.html')
    return fig
