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
import json as _j
_p = os.path.join(config.WORK_DIR, "dopt_sets.json")
if os.path.exists(_p):
    _d = _j.load(open(_p))
    ok = np.allclose(sorted(_d["opt6"]), sorted(OPT6F)) and np.allclose(sorted(_d["opt12"]), sorted(OPT12F))
    print("D-optimal kümeleri optdesign.py ile uyumlu (doğrulandı)" if ok else "UYARI: optdesign.py D-optimal kümeleri bu dosyadaki OPT6F/OPT12F listelerinden farklı!", flush=True)
CFG = {"full1": (np.arange(111), False), "dopt12": (sel(OPT12F), False), "uni12": (sel(np.arange(0.5, 6.01, 0.5)), False),
       "uni6": (sel([0.5,1,2,3,4,6]), False), "dopt6": (sel(OPT6F), False), "full2": (np.arange(111), True)}
EXT_CFG = ["full1", "dopt12", "uni12", "full2"]
T1, T2, TY, TD = gen(config.n(24, 3), 71); R1, R2, RY, RD = gen(config.n(400, 60), 72); X1, X2, XY, XD, XG = real_test(config.n(4, 1), 81)
print(f"sentetik test n={len(TD)}; SVR eğitim n={len(RD)}; gerçek-türevli test n={len(XD)} ({XG.max()+1} kayıt)", flush=True)
err = lambda P, Y: np.mean(np.abs(P[:, :12]-Y)/np.abs(Y)*100, axis=1)          # spektrum başına ort. APE (12 hedef)
E = {}; PRED = {}
for name, (fi, two) in CFG.items():
    tf = lambda a, b: np.log10(np.maximum(np.hstack([a[:, fi]] + ([b[:, fi]] if two else [])), 1e-6))
    tgt = np.column_stack([RY, RD*1000]); sc = StandardScaler().fit(tf(R1, R2)); ys = StandardScaler().fit(tgt)
    m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tf(R1, R2)), ys.transform(tgt))
    pr = lambda a, b: ys.inverse_transform(m.predict(sc.transform(tf(a, b))))[:, :13]
    PRED[("syn", "SVR", name)] = pr(T1, T2); E[("syn", "SVR", name)] = err(PRED[("syn", "SVR", name)], TY)
    if name in EXT_CFG: PRED[("ext", "SVR", name)] = pr(X1, X2); E[("ext", "SVR", name)] = err(PRED[("ext", "SVR", name)], XY)
    print(f"SVR {name} hazır", flush=True)
for name, (fi, two) in CFG.items():
    t = time.time(); PRED[("syn", "NLLS", name)] = np.array([nlls(T1[i], T2[i], fi, two) for i in range(len(TD))]); E[("syn", "NLLS", name)] = err(PRED[("syn", "NLLS", name)], TY)
    if name in EXT_CFG: PRED[("ext", "NLLS", name)] = np.array([nlls(X1[i], X2[i], fi, two) for i in range(len(XD))]); E[("ext", "NLLS", name)] = err(PRED[("ext", "NLLS", name)], XY)
    np.savez(f"{OUT}/preds.npz", **{"|".join(k): v for k, v in PRED.items()}, TY=TY, TD=TD, XY=XY, XD=XD, ext_groups=XG)
    print(f"NLLS {name} bitti ({time.time()-t:.0f}s)", flush=True)
# ---- istatistik ----
def holm(p):
    p = np.asarray(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p)): run = max(run, (m - r)*p[i]); adj[i] = min(1.0, run)
    return adj
def paired(a, b, groups=None, B=4000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); ga, gb = a > 10, b > 10
    if groups is None: bi = [rng.integers(0, n, n) for _ in range(B)]
    else:
        cl = np.unique(groups); mem = {c: np.where(groups == c)[0] for c in cl}
        bi = [np.concatenate([mem[c] for c in rng.choice(cl, len(cl))]) for _ in range(B)]
    dm = np.array([np.median(a[i]) - np.median(b[i]) for i in bi]); dg = np.array([(ga[i].mean() - gb[i].mean())*100 for i in bi])
    n01, n10 = int(np.sum(ga & ~gb)), int(np.sum(~ga & gb))
    p_m = binomtest(min(n01, n10), n01+n10, .5).pvalue if n01+n10 else 1.0
    try: p_w = wilcoxon(a - b).pvalue
    except ValueError: p_w = 1.0
    return {"d_med": float(np.median(a) - np.median(b)), "d_med_ci": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            "d_gross_pp": float((ga.mean() - gb.mean())*100), "d_gross_ci": [float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5))],
            "p_mcnemar": float(p_m), "p_wilcoxon": float(p_w), "gross_a": float(ga.mean()*100), "gross_b": float(gb.mean()*100), "med_a": float(np.median(a)), "med_b": float(np.median(b))}
COMP = [("NLLS vs SVR, full band", ("NLLS", "full1"), ("SVR", "full1")), ("NLLS vs SVR, D-opt 12", ("NLLS", "dopt12"), ("SVR", "dopt12")),
        ("NLLS: uniform 12 vs D-opt 12", ("NLLS", "uni12"), ("NLLS", "dopt12")), ("SVR: uniform 12 vs D-opt 12", ("SVR", "uni12"), ("SVR", "dopt12")),
        ("NLLS: uniform 6 vs D-opt 6", ("NLLS", "uni6"), ("NLLS", "dopt6")), ("SVR: uniform 6 vs D-opt 6", ("SVR", "uni6"), ("SVR", "dopt6")),
        ("NLLS: 1 vs 2 thicknesses (full)", ("NLLS", "full1"), ("NLLS", "full2")), ("SVR: 1 vs 2 thicknesses (full)", ("SVR", "full1"), ("SVR", "full2"))]
res = {}
for setn, grp in [("syn", None), ("ext", XG)]:
    rows = []
    for title, (ma, ca), (mb, cb) in COMP:
        if (setn, ma, ca) in E and (setn, mb, cb) in E: rows.append((title, paired(E[(setn, ma, ca)], E[(setn, mb, cb)], grp)))
    ph = holm([r[1]["p_mcnemar"] for r in rows]); pw = holm([r[1]["p_wilcoxon"] for r in rows])
    for (t, r), a, b_ in zip(rows, ph, pw): r["p_mcnemar_holm"] = float(a); r["p_wilcoxon_holm"] = float(b_)
    res[setn] = {t: r for t, r in rows}
    print(f"\n=== {'SENTETİK (n=%d, spektrum bootstrap)' % len(TD) if setn == 'syn' else 'GERÇEK-TÜREVLİ (%d kayıt, küme bootstrap)' % (XG.max()+1)} ===  Δ = (ilk) − (ikinci); pozitif = ilk daha kötü", flush=True)
    for t, r in rows:
        print(f"  {t:34s} Δmedyan APE {r['d_med']:+6.2f} [{r['d_med_ci'][0]:+6.2f},{r['d_med_ci'][1]:+6.2f}] | Δaşırı hata {r['d_gross_pp']:+6.1f} pp [{r['d_gross_ci'][0]:+6.1f},{r['d_gross_ci'][1]:+6.1f}] | McNemar p_Holm {r['p_mcnemar_holm']:.2g} | Wilcoxon p_Holm {r['p_wilcoxon_holm']:.2g}", flush=True)
json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1)
print("\nBİTTİ", flush=True)
