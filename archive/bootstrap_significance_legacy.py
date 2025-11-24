# bootstrap_significance.py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CSV   = "results/nowcast_scores.csv"
N_BOOT = 5000
SEED   = 42
rng    = np.random.default_rng(SEED)

# --- Load and derive features ---
df = pd.read_csv(CSV)

# derive the intended columns
df["absS"]     = df["S"].abs()
df["inv_relax"] = 1.0 / (df["relax"] + 1e-9)

features = [
    ("absS",        "+"),
    ("inv_relax",   "+"),
    ("agree",       "+"),
    ("dir_var",     "~"),   # near 0 expected
    ("shear_var",   "-"),
]

def corr_pair(a, b, method="pearson"):
    if method == "pearson":
        r, p = pearsonr(a, b)
    else:
        r, p = spearmanr(a, b)
    return float(r), float(p)

rows = []
for name, sign in features:
    rP, pP = corr_pair(df["risk"], df[name], "pearson")
    rS, pS = corr_pair(df["risk"], df[name], "spearman")

    # bootstrap Pearson r
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(df), size=len(df))
        rr, _ = corr_pair(df["risk"].to_numpy()[idx], df[name].to_numpy()[idx], "pearson")
        boots.append(rr)
    boots = np.asarray(boots)
    mu, sd = float(boots.mean()), float(boots.std(ddof=1))
    z = (rP - mu) / sd if sd > 0 else np.nan
    p_boot = float((np.abs(boots) >= abs(rP)).mean())

    rows.append({
        "feature": name,
        "expected_sign": sign,
        "r_pearson": rP, "p_pearson": pP,
        "r_spearman": rS, "p_spearman": pS,
        "boot_mean_r": mu, "boot_sd_r": sd,
        "z_score": z, "p_boot": p_boot
    })

res = pd.DataFrame(rows)
res = res[["feature","expected_sign","r_pearson","p_pearson","r_spearman","p_spearman","boot_mean_r","boot_sd_r","z_score","p_boot"]]
out = "results/bootstrap_significance.csv"
res.to_csv(out, index=False)
print(res.to_string(index=False))
print(f"\nSaved → {out}")