import config  # ortak yollar ve SMOKE modu
# Birleşik tasarım deneyi + istatistiksel analiz (eşleştirilmiş bootstrap GA, McNemar, Wilcoxon, Holm)
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from loadreal import load_all, at_freqs
from scipy.optimize import least_squares
from scipy.stats import binomtest, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz; DELTA = 0.010; OUT = config.STATS_DIR; os.makedirs(OUT, exist_ok=True)
def s21m(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def gen(n_per, seed):
    rng = np.random.default_rng(seed); S1, S2, Y, D = [], [], [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); d1 = rng.uniform(0.010, 0.040, n_per)
        S1.append(g.add_noise(s21m(e, d1[:, None]), rng, 45, 0.02)); S2.append(g.add_noise(s21m(e, (d1+DELTA)[:, None]), rng, 45, 0.02)); D.append(d1)
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S1), np.vstack(S2), np.vstack(Y), np.concatenate(D)
KEYS = ["Heart_standard","KidneyCortex_standard","Spleen_standard","Liver_standard","SkeletalMuscle_standard","BrainGreyMatter_standard"]
RD_ = load_all(config.REAL_DIR)
def real_test(n_draw, seed):
    rng = np.random.default_rng(seed); E, Y, D1, G = [], [], [], []; lt = np.log(F*1e9); rid = 0
    for k in KEYS:
        v = RD_[k]; lf = np.log(v["f"]); er6, sg6 = at_freqs(v, [0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
        for j in range(v["eps_re"].shape[1]):
            er = np.interp(lt, lf, v["eps_re"][:, j]); ei = np.interp(lt, lf, v["eps_im"][:, j]); y = np.column_stack([er6[j], sg6[j]]).ravel()
            for _ in range(n_draw): E.append(er - 1j*ei); Y.append(y); D1.append(rng.uniform(0.010, 0.040)); G.append(rid)
            rid += 1
    E, Y, D1 = np.array(E), np.array(Y), np.array(D1)
    return g.add_noise(s21m(E, D1[:, None]), rng, 45, 0.02), g.add_noise(s21m(E, (D1+DELTA)[:, None]), rng, 45, 0.02), Y, D1, np.array(G)
def eps_model(p):
    ef, de, tps, a, sg = p[:5]; return ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
lo, hi = [1,1,3,0,0,0.005], [10,90,20,.5,3.0,0.045]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def nlls(S1, S2, fi, two):
    best = None
    for d0 in [0.015, 0.025, 0.035]:
        for x0 in st5:
            def f(p):
                e = eps_model(p); r = [s21m(e, p[5])[fi] - S1[fi]]
                if two: r.append(s21m(e, p[5]+DELTA)[fi] - S2[fi])
                return np.concatenate(r)
            r = least_squares(f, x0+[d0], bounds=(lo, hi), x_scale=[1,10,3,.1,.5,.01])
            if best is None or r.cost < best.cost: best = r
    e = eps_model(best.x); return np.concatenate([np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), [best.x[5]*1000]])
sel = lambda fl: np.array([int(np.argmin(abs(F-x))) for x in fl])
OPT6F = [0.5, 0.85, 1.25, 1.7, 4.4, 6.0]; OPT12F = [0.5, 0.7, 0.85, 1.0, 1.25, 1.7, 2.35, 2.85, 3.6, 4.4, 5.95, 6.0]
CFG = {"full1": (np.arange(111), False), "dopt12": (sel(OPT12F), False), "uni12": (sel(np.arange(0.5, 6.01, 0.5)), False),
       "uni6": (sel([0.5,1,2,3,4,6]), False), "dopt6": (sel(OPT6F), False), "full2": (np.arange(111), True)}
EXT_CFG = ["full1", "dopt12", "uni12", "full2"]

from sklearn.neural_network import MLPRegressor
pre = np.load(f"{OUT}/preds.npz")
T1, T2, TY, TD = gen(config.n(24, 3), 71); R1, R2, RY, RD = gen(config.n(400, 60), 72); X1, X2, XY, XD, XG = real_test(config.n(4, 1), 81)
assert np.allclose(TY, pre["TY"]) and np.allclose(XY, pre["XY"]) and np.allclose(XG, pre["ext_groups"]), "test setleri stats_design ile aynı değil"
print("test setleri stats_design ile birebir aynı (doğrulandı)", flush=True)
err = lambda P, Y: np.mean(np.abs(P[:, :12]-Y)/np.abs(Y)*100, axis=1)
PM = {}
for name in ["full1", "dopt12", "uni12", "full2"]:
    fi, two = CFG[name]
    tf = lambda a, b: np.log10(np.maximum(np.hstack([a[:, fi]] + ([b[:, fi]] if two else [])), 1e-6))
    tgt = np.column_stack([RY, RD*1000]); sc = StandardScaler().fit(tf(R1, R2)); ys = StandardScaler().fit(tgt)
    for seed in (0, 1, 2):
        t = time.time(); m = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-4, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, max_iter=600, random_state=seed).fit(sc.transform(tf(R1, R2)), ys.transform(tgt))
        pr = lambda a, b: ys.inverse_transform(m.predict(sc.transform(tf(a, b))))[:, :13]
        PM[("syn", f"MLP{seed}", name)] = pr(T1, T2); PM[("ext", f"MLP{seed}", name)] = pr(X1, X2)
        print(f"MLP {name} seed{seed} ({time.time()-t:.0f}s, {m.n_iter_} iter)", flush=True)
    np.savez(f"{OUT}/preds_mlp.npz", **{"|".join(k): v for k, v in PM.items()})
def met(p, Y, D): A = np.abs(p[:, :12]-Y)/np.abs(Y)*100; return np.median(A[:, 0::2]), np.median(A[:, 1::2]), np.mean(A.mean(1) > 10)*100, np.sqrt(np.mean((p[:, 12]-D*1000)**2))
print("\nMLP vs SVR vs NLLS (aynı test setleri); MLP: 3 tohumun ortalaması [min–maks aşırı hata]")
for setn, Y, D in [("syn", TY, TD), ("ext", XY, XD)]:
    print(f"\n[{'SİMÜLASYON n=%d' % len(D) if setn == 'syn' else 'GERÇEK-TÜREVLİ n=%d' % len(D)}]  (ε′/σ medyan APE, aşırı hata %, d RMSE mm)")
    for name in ["full1", "dopt12", "uni12", "full2"]:
        if (setn, "MLP0", name) not in PM: continue
        ms = np.array([met(PM[(setn, f"MLP{s}", name)], Y, D) for s in (0, 1, 2)]); mm = ms.mean(0)
        sv = met(pre[f"{setn}|SVR|{name}"], Y, D); nl = met(pre[f"{setn}|NLLS|{name}"], Y, D)
        print(f"  {name:7s} MLP {mm[0]:.1f}/{mm[1]:.1f}, {mm[2]:.0f}% [{ms[:,2].min():.0f}–{ms[:,2].max():.0f}], d {mm[3]:.1f} | SVR {sv[0]:.1f}/{sv[1]:.1f}, {sv[2]:.0f}%, d {sv[3]:.1f} | NLLS {nl[0]:.1f}/{nl[1]:.1f}, {nl[2]:.0f}%, d {nl[3]:.1f}", flush=True)
# eşleştirilmiş: MLP(tohum 0) − SVR
def paired(a, b, groups=None, B=3000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); ga, gb = a > 10, b > 10
    if groups is None: bi = [rng.integers(0, n, n) for _ in range(B)]
    else:
        cl = np.unique(groups); mem = {c: np.where(groups == c)[0] for c in cl}; bi = [np.concatenate([mem[c] for c in rng.choice(cl, len(cl))]) for _ in range(B)]
    dm = np.array([np.median(a[i]) - np.median(b[i]) for i in bi]); dg = np.array([(ga[i].mean() - gb[i].mean())*100 for i in bi])
    n01, n10 = int(np.sum(ga & ~gb)), int(np.sum(~ga & gb)); p = binomtest(min(n01, n10), n01+n10, .5).pvalue if n01+n10 else 1.0
    return np.median(a) - np.median(b), np.percentile(dm, [2.5, 97.5]), (ga.mean() - gb.mean())*100, np.percentile(dg, [2.5, 97.5]), p
print("\nEşleştirilmiş fark: MLP (tohum 0) − SVR; pozitif = MLP daha kötü")
for setn, grp in [("syn", None), ("ext", XG)]:
    for name in ["full1", "dopt12", "uni12", "full2"]:
        dmed, ci, dg, cg, p = paired(err(PM[(setn, "MLP0", name)], TY if setn == "syn" else XY), err(pre[f"{setn}|SVR|{name}"], TY if setn == "syn" else XY), grp)
        print(f"  {setn} {name:7s} Δmedyan {dmed:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | Δaşırı hata {dg:+.1f} pp [{cg[0]:+.1f},{cg[1]:+.1f}] | McNemar p (düzeltmesiz) {p:.3g}", flush=True)
print("BİTTİ", flush=True)
