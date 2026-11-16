import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, sys, time, warnings
warnings.filterwarnings('ignore'); pass
from scipy.optimize import least_squares
from loadreal import *
c0, e0, d = 3e8, 8.854e-12, 0.03; FREQ = np.linspace(.5, 6, 111); W = 2*np.pi*FREQ*1e9
FT = [0.5,1,2,3,4,6]; IDX = [0,10,30,50,70,110]
def model_eps(p): ef, de, tps, a, sg = p; return ef + de/(1+(1j*W*tps*1e-12)**(1-a)) - 1j*sg/(W*e0)
def s21(e):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(W/c0)*d); G = (1-n)/(1+n); return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def noise(S, rng, snr=45, mult=0.02):
    pw = np.mean(S**2); Sn = np.clip(S + rng.normal(0, np.sqrt(pw/10**(snr/10)), S.shape), 1e-6, 1); return np.clip(Sn*rng.normal(1, mult, S.shape), 1e-6, 1)
lo, hi = [1,1,3,0,0], [10,90,20,.5,3.0]; starts = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1],[3,10,8,.1,.1]]
D = load_all(config.REAL_DIR); rng = np.random.default_rng(0); rows = []; tm = []
for key, v in D.items():
    if not key.endswith("_standard"): continue
    lf, lt = np.log(v["f"]), np.log(FREQ*1e9)
    er = np.array([np.interp(lt, lf, v["eps_re"][:, j]) for j in range(v["eps_re"].shape[1])]); ei = np.array([np.interp(lt, lf, v["eps_im"][:, j]) for j in range(v["eps_im"].shape[1])])
    e_t, s_t = at_freqs(v, FT)
    for i in range(er.shape[0]):
        S0 = s21(er[i] - 1j*ei[i]); obs = noise(S0, rng); t = time.perf_counter()
        b = min((least_squares(lambda p: s21(model_eps(p)) - obs, x0, bounds=(lo, hi), x_scale=[1,10,3,.1,.5]) for x0 in starts), key=lambda r: r.cost)
        tm.append((time.perf_counter()-t)*1000); e = model_eps(b.x)
        pred = {"ε′": e.real[IDX], "σ_eff": -e.imag[IDX]*W[IDX]*e0}; true = {"ε′": e_t[i], "σ_eff": s_t[i]}
        for q in pred:
            for j, f in enumerate(FT): rows.append({"tissue": key.replace("_standard",""), "q": q, "f": f, "ape": abs(pred[q][j]-true[q][j])/abs(true[q][j])*100})
r = pd.DataFrame(rows); r.to_csv(config.WORK_DIR + "/nlls_real_rows.csv", index=False); soft = ["Heart","KidneyCortex","Spleen","Liver","SkeletalMuscle","BrainGreyMatter"]
print(f"Klasik NLLS (1 kutup+σ, 4 başlangıç) — GERÇEK-türevli S21 (+gürültü), {len(tm)} spektrum, ort. {np.mean(tm):.0f} ms/spektrum (sandbox, 1 çekirdek)")
print("\nYumuşak organlar (6 doku), medyan APE [%]  [karşılaştırma: SVR ε′ 0.8-1.3, σ_eff 0.6-1.8]:")
print(r[r.tissue.isin(soft)].pivot_table(index="q", columns="f", values="ape", aggfunc="median").round(1).to_string())
print("\nTüm dokular (14), medyan APE [%]:"); print(r.pivot_table(index="q", columns="f", values="ape", aggfunc="median").round(1).to_string())
print("\nDoku başına medyan APE:"); print(r.groupby("tissue").ape.median().round(1).sort_values().to_string())
