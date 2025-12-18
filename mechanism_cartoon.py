import matplotlib.pyplot as plt
from pathlib import Path

def plot_mechanism(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.text(0.5, 0.5, "Mechanism Cartoon Placeholder", ha='center', va='center', fontsize=14)
    ax.axis('off')
    fig.savefig(output_dir/'mechanism_cartoon.svg', bbox_inches='tight')
    fig.savefig(output_dir/'mechanism_cartoon.png', dpi=300, bbox_inches='tight')
    return fig
