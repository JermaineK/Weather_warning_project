import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def hist_parity(df, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4,3))
    plt.hist(df['S'].dropna(), bins=40)
    plt.xlabel('Spiral index S'); plt.ylabel('Count'); plt.title('Parity distribution')
    plt.tight_layout(); plt.savefig(Path(out_dir)/'hist_S.png', dpi=160); plt.close()

def scatter_relax_S(df, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4,3))
    plt.scatter(df['relax'], np.abs(df['S']), s=6, alpha=0.4)
    plt.xlabel('Relaxation ratio'); plt.ylabel('|S|'); plt.title('Coherence vs mobility')
    plt.tight_layout(); plt.savefig(Path(out_dir)/'scatter_relax_S.png', dpi=160); plt.close()
