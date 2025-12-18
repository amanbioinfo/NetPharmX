import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_chem_similarity(targets_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(targets_csv)
    # dummy similarity
    df['ChemSimilarity'] = 0.5
    df['TargetOverlap'] = 1
    fig, ax = plt.subplots()
    ax.scatter(df['ChemSimilarity'], df['TargetOverlap'], c='#9467bd', s=50)
    ax.set_xlabel('Chemical Similarity')
    ax.set_ylabel('Target Overlap')
    ax.set_title('Chemical Similarity vs Target Overlap')
    plt.tight_layout()
    fig.savefig(output_dir/'chem_similarity_vs_target_overlap.svg', bbox_inches='tight')
    fig.savefig(output_dir/'chem_similarity_vs_target_overlap.png', dpi=300, bbox_inches='tight')
    return fig
