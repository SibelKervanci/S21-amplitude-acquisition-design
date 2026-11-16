import config  # shared paths and SMOKE mode
# envelope_check.py — do the measured records fall inside the simulated class distributions? (Table VI, columns 3-6)
# For each mapped tissue and each of the 12 measurands (eps' and sigma_eff at 0.5, 1, 2, 3, 4, 6 GHz):
#   "inside" = share of measured records within the 1-99 % range of the simulated class; "ratio" = median(measured) / median(simulated).
# The table reports the mean over the six frequencies, separately for eps' and sigma_eff.
import numpy as np, pandas as pd
from loadreal import load_all, at_freqs, MAP
FR = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
D = load_all(config.REAL_DIR); syn = pd.read_csv(config.TRAIN_CSV); rows = []
for rk, cls in MAP.items():
    er, sg = at_freqs(D[rk], FR); s = syn[syn.tissue == cls]
    for kind, T, col0 in (("eps", er, "epsr"), ("sig", sg, "sigeff")):
        for j, f in enumerate(FR):
            col = s[f"{col0}_f{str(f).replace('.', 'p')}"].values; lo, hi = np.percentile(col, [1, 99])
            rows.append(dict(cls=cls, kind=kind, f=f, n=len(T), inside=float(np.mean((T[:, j] >= lo) & (T[:, j] <= hi))), ratio=float(np.median(T[:, j]) / np.median(col))))
e = pd.DataFrame(rows); e.to_csv(config.WORK_DIR + "/real_envelope.csv", index=False)
print("Measured records vs simulated classes (mean over the six frequencies)")
print(f"{'class':12s} {'n':>3s} | {'eps inside':>10s} {'sigma inside':>12s} | {'eps ratio':>9s} {'sigma ratio':>11s}")
for cls in MAP.values():
    a = e[(e.cls == cls) & (e.kind == "eps")]; b = e[(e.cls == cls) & (e.kind == "sig")]
    print(f"{cls:12s} {a.n.iloc[0]:3d} | {a.inside.mean():10.0%} {b.inside.mean():12.0%} | {a.ratio.mean():9.2f} {b.ratio.mean():11.2f}")
