import argparse, pandas as pd, numpy as np
from pathlib import Path
from .spiral_filters import spiral_index, vorticity_divergence, relaxation_ratio, parity_agreement

def load_tile_arrays(row):
    u = np.array(row['u_grid']).reshape(row['nlat'], row['nlon'])
    v = np.array(row['v_grid']).reshape(row['nlat'], row['nlon'])
    lat = np.array(row['lat_vec'])
    lon = np.array(row['lon_vec'])
    return u, v, lat, lon

def build_features(windows_path: str, out_path: str):
    df = pd.read_parquet(windows_path)
    feats = []
    for _, r in df.iterrows():
        u, v, lat, lon = load_tile_arrays(r)
        S = spiral_index(u, v)
        zeta, div = vorticity_divergence(u, v, lat, lon)
        zeta_mean = float(np.nanmean(zeta))
        relax = relaxation_ratio(zeta)
        agree = parity_agreement(S, zeta_mean)
        feats.append({
            'storm': r.get('storm', ''),
            'time': r['time'],
            'lat_c': r['lat_c'],
            'lon_c': r['lon_c'],
            'S': S,
            'zeta_mean': zeta_mean,
            'div_mean': float(np.nanmean(div)),
            'relax': relax,
            'agree': agree,
            'label_genesis24': r.get('label_genesis24', np.nan),
            'label_intensity_up24': r.get('label_intensity_up24', np.nan),
        })
    pd.DataFrame(feats).to_parquet(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    build_features(args.windows, args.out)

if __name__ == '__main__':
    main()
