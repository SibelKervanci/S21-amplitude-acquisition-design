import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings('ignore'); pass
from loadreal import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
D = load_all(config.REAL_DIR); FT = [0.5,1,2,3,4,6]
def feats(key):
    er, sg = at_freqs(D[key], FT); return np.column_stack([np.column_stack([er[:, j], sg[:, j]]) for j in range(6)])
# (a) GERÇEK veride 7 sınıfın kendi içinde ayrılabilirliği (leave-one-out, 12 özellik, lojistik regresyon)
Xr = np.vstack([feats(k) for k in MAP]); yr = np.concatenate([[c]*D[k]['eps_re'].shape[1] for k, c in MAP.items()])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
pred = cross_val_predict(clf, Xr, yr, cv=LeaveOneOut())
print(f"(a) GERÇEK ölçümler, 7 sınıf, leave-one-out doğruluk = {np.mean(pred==yr):.3f}  (n={len(yr)}; şans ≈ {1/7:.2f}, çoğunluk sınıfı = {max(pd.Series(yr).value_counts())/len(yr):.2f})")
cm = pd.crosstab(pd.Series(yr, name="gerçek"), pd.Series(pred, name="tahmin")); print(cm.to_string())
# (b) Heart/Kidney/Spleen/Muscle: gerçek verideki etki büyüklüğü (Cohen d) — sentetik d ile karşılaştır
def d(x, z): return abs(x.mean()-z.mean())/np.sqrt((x.std(ddof=1)**2+z.std(ddof=1)**2)/2)
print("\n(b) Cohen d, GERÇEK ölçümler (σ_eff @1 GHz ve ε′ @1 GHz):   [sentetik d: σ_eff H-K 0.83 H-S 0.19 K-S 0.85 | ε′ H-K 0.20 H-S 0.55 K-S 0.36]")
for nm, j in [("σ_eff@1GHz", 3), ("ε′@1GHz", 2)]:
    v = {c: feats(k)[:, j] for k, c in MAP.items() if c in ("Heart","Kidney","Spleen","Muscle")}
    print(f"  {nm:11s} H-K={d(v['Heart'],v['Kidney']):.2f}  H-S={d(v['Heart'],v['Spleen']):.2f}  K-S={d(v['Kidney'],v['Spleen']):.2f}  H-M={d(v['Heart'],v['Muscle']):.2f}  | gerçek medyanlar: " + ", ".join(f"{c} {np.median(x):.2f}" for c, x in v.items()))
# (c) Kaba grup düzeyi: gerçek spektrumların S21'i -> sentetik-eğitimli 17 sınıflı model -> tahmin, fiziksel gruba düşüyor mu?
exec(open(config.ROOT + '/semireal.py', encoding='utf-8').read().split("K = 20; rng")[0].split("D = load_all")[0])
