import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, sys, warnings, time
warnings.filterwarnings('ignore'); pass
from loadreal import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
c0, d = 3e8, 0.03; FREQ = np.linspace(.5, 6, 111); W = 2*np.pi*FREQ*1e9
def s21_from_eps(e):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(W/c0)*d); G = (1-n)/(1+n); return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def add_noise(S, rng, snr=45, mult=0.02):
    pw = np.mean(S**2, axis=1, keepdims=True); Sn = np.clip(S + rng.normal(0, 1, S.shape)*np.sqrt(pw/10**(snr/10)), 1e-6, 1.0); return np.clip(Sn*rng.normal(1.0, mult, S.shape), 1e-6, 1.0)
D = load_all(config.REAL_DIR); FT = [0.5,1,2,3,4,6]; TAG = ["0p5","1p0","2p0","3p0","4p0","6p0"]
tr = pd.read_csv(config.TRAIN_CSV)
if config.SMOKE: tr = tr.sample(2500, random_state=0).reset_index(drop=True)
S = [c for c in tr.columns if c.startswith('s21_')]
cols = [f"{k}_f{t}" for t in TAG for k in ("epsr","sigeff")]
sc = StandardScaler().fit(tr[S].values); ysc = StandardScaler().fit(tr[cols].values)
t0 = time.time(); svr = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tr[S].values), ysc.transform(tr[cols].values)); print(f"SVR eğitim {time.time()-t0:.0f}s (17.000 sentetik spektrum, 12 hedef)")
rng = np.random.default_rng(0); K = 5; rows = []
for key, v in D.items():
    if not key.endswith("_standard"): continue
    lf, lt = np.log(v["f"]), np.log(FREQ*1e9)
    er = np.array([np.interp(lt, lf, v["eps_re"][:, j]) for j in range(v["eps_re"].shape[1])]); ei = np.array([np.interp(lt, lf, v["eps_im"][:, j]) for j in range(v["eps_im"].shape[1])])
    Z = add_noise(np.repeat(s21_from_eps(er - 1j*ei), K, axis=0), rng)
    P = ysc.inverse_transform(svr.predict(sc.transform(Z))).reshape(er.shape[0], K, 12).mean(1)
    e_t, s_t = at_freqs(v, FT); T = np.column_stack([np.column_stack([e_t[:, j], s_t[:, j]]) for j in range(6)])
    for i in range(T.shape[0]):
        for j in range(12): rows.append({"tissue": key.replace("_standard",""), "j": j, "f": FT[j//2], "q": "ε′" if j%2==0 else "σ_eff", "true": T[i, j], "pred": P[i, j]})
r = pd.DataFrame(rows); r["ape"] = (r.pred-r.true).abs()/r.true.abs()*100
print(f"\nGERÇEK insan dokusu ({r.tissue.nunique()} doku, {len(r)//12} örnek) -> S21 -> sentetik-eğitimli SVR -> ε′, σ_eff   (medyan mutlak yüzde hata)")
print(r.pivot_table(index="q", columns="f", values="ape", aggfunc="median").round(1).to_string())
soft = r[r.tissue.isin(["Heart","KidneyCortex","Spleen","Liver","SkeletalMuscle","BrainGreyMatter"])]
print("\nYalnızca yumuşak organlar (Heart, KidneyCortex, Spleen, Liver, SkeletalMuscle, BrainGrey):"); print(soft.pivot_table(index="q", columns="f", values="ape", aggfunc="median").round(1).to_string())
print("\nDoku başına medyan yüzde hata (12 hedef ort.):"); print(r.groupby("tissue").ape.median().round(1).sort_values().to_string())
r.to_csv(config.WORK_DIR + "/reg_real_rows.csv", index=False)
print("\nBirleşik R² (tüm dokular birlikte; doku-arası varyans yüksek olduğu için iyimser) ve yumuşak-organ-içi R²:")
for q in ("ε′", "σ_eff"):
    a = r[r.q == q]; b = soft[soft.q == q]
    r2 = lambda x: 1-((x.true-x.pred)**2).sum()/((x.true-x.true.mean())**2).sum()
    print(f"  {q:6s} birleşik R²={r2(a):.3f}   yumuşak-organ-içi R²={r2(b):.3f}")
