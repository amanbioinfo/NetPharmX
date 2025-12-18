import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_disease_bubble(disease_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(disease_csv)
    df_grouped = df.groupby('Disease').agg({'Target':'count','Compound':'nunique'}).reset_index()

    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(df_grouped['Disease'], df_grouped['Target'], s=df_grouped['Compound']*50, alpha=0.7, color='#d62728')
    ax.set_xlabel('Disease')
    ax.set_ylabel('Number of Targets')
    ax.set_title('Disease Class Bubble Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(output_dir/'disease_class_bubble.svg', bbox_inches='tight')
    fig.savefig(output_dir/'disease_class_bubble.png', dpi=300, bbox_inches='tight')
    return fig
