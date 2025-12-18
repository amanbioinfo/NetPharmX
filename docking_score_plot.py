import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_docking_scores(docking_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    try:
        df = pd.read_csv(docking_csv)
    except:
        df = pd.DataFrame({'Compound':['A'], 'Score':[0]})
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df['Compound'], df['Score'], color='#ff7f0e')
    ax.set_title('Docking Score Distribution')
    plt.tight_layout()
    fig.savefig(output_dir/'docking_score_plot.svg', bbox_inches='tight')
    fig.savefig(output_dir/'docking_score_plot.png', dpi=300, bbox_inches='tight')
    return fig
