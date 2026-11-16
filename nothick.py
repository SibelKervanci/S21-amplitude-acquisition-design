import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
IDX = g.TARGET_IDX; FT = g.TARGET_F; OM = g.OMEGA; c0, e0 = g.c, g.e0
def gen(n_per, seed, tmin=0.010, tmax=0.050):
    rng = np.random.default_rng(seed); S_all=[]; T_all=[]; D_all=[]
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); d = rng.uniform(tmin, tmax, n_per); e = g.eps_4pole(P)
        S_all.append(g.add_noise(g.s21_slab(e, d), rng, 45, 0.02)); D_all.append(d)
        T_all.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S_all), np.vstack(T_all), np.concatenate(D_all)
Xtr, Ytr, Dtr = gen(config.n(1000, 100), 11); Xte, Yte, Dte = gen(config.n(18, 3), 22)
sub = np.random.default_rng(0).choice(len(Xtr), min(8000, len(Xtr)), replace=False); Xs, Ys, Ds = Xtr[sub], Ytr[sub], Dtr[sub]
np.savez(config.WORK_DIR + '/nothick_data.npz', Xte=Xte, Yte=Yte, Dte=Dte)
print(f"eğitim {Xtr.shape}, test {Xte.shape}; kalınlık U(10,50) mm; medyan |S21|: 0.5GHz={np.median(Xtr[:,0]):.3f}, 3GHz={np.median(Xtr[:,50]):.4f}, 6GHz={np.median(Xtr[:,110]):.5f}")
def ape(P, T): return np.abs(P-T)/np.abs(T)*100
res = {}
for name, tf in [("SVR log-genlik", lambda X: np.log10(np.maximum(X, 1e-6))), ("SVR genlik", lambda X: X)]:
    sc = StandardScaler().fit(tf(Xs)); ys = StandardScaler().fit(np.column_stack([Ys, Ds*1000]))
    t = time.time(); m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tf(Xs)), ys.transform(np.column_stack([Ys, Ds*1000])))
    out = ys.inverse_transform(m.predict(sc.transform(tf(Xte)))); res[name] = (out[:, :12], out[:, 12]/1000); print(f"{name}: eğitim {time.time()-t:.0f}s (8000 örnek)", flush=True); np.savez(f'{config.WORK_DIR}/nothick_{name.replace(" ","_")}.npz', P=res[name][0], D=res[name][1])
def mdl(p, d):
    ef, de, tps, a, sg = p; e = ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return e, np.abs(((1-G**2)*z)/(1-G**2*z**2))
lo5, hi5 = [1,1,3,0,0], [10,90,20,.5,3.0]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def nlls(obs, free_d, d_known=None, log=False):
    best = None
    ds = [0.02, 0.03, 0.04] if free_d else [d_known]
    for d0 in ds:
        for x0 in st5:
            if free_d: f = lambda p: (np.log10(np.maximum(mdl(p[:5], p[5])[1], 1e-6)) - np.log10(np.maximum(obs, 1e-6))) if log else (mdl(p[:5], p[5])[1] - obs); lo, hi, x = lo5+[0.005], hi5+[0.06], x0+[d0]; xs = [1,10,3,.1,.5,.01]
            else: f = lambda p: (mdl(p, d_known)[1] - obs); lo, hi, x = lo5, hi5, x0; xs = [1,10,3,.1,.5]
            r = least_squares(f, x, bounds=(lo, hi), x_scale=xs)
            if best is None or r.cost < best.cost: best = r
    e = mdl(best.x[:5], best.x[5] if free_d else d_known)[0]
    return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), (best.x[5] if free_d else d_known)
for label, kw in [("NLLS, kalınlık SERBEST", dict(free_d=True)), ("NLLS, kalınlık BİLİNİYOR (oracle)", dict(free_d=False))]:
    P = []; Dd = []; t = time.time()
    for i in range(len(Xte)):
        p, dd = nlls(Xte[i], d_known=Dte[i], **kw); P.append(p); Dd.append(dd)
    res[label] = (np.array(P), np.array(Dd)); print(f"{label}: {(time.time()-t)/len(Xte)*1000:.0f} ms/spektrum", flush=True); np.savez(config.WORK_DIR+"/nothick_"+label.split(",")[1].strip().split()[0]+".npz", P=res[label][0], D=res[label][1])
print(f"\nKALINLIK BİLİNMİYOR (10-50 mm), test n={len(Xte)} — medyan mutlak yüzde hata [%] (ε′ | σ_eff, 6 frekans ortalaması), %10'dan kötü olan örnek oranı, kalınlık RMSE [mm]")
print(f"{'yöntem':46s} {'ε′ APE':>7} {'σ_eff APE':>10} {'>%10 hata':>10} {'d RMSE mm':>10}")
for k, (P, Dp) in res.items():
    A = ape(P, Yte); e_, s_ = A[:, 0::2], A[:, 1::2]
    drm = "-" if k.endswith("oracle)") else f"{np.sqrt(np.mean((Dp-Dte)**2))*1000:.1f}"
    print(f"{k:46s} {np.median(e_):7.1f} {np.median(s_):10.1f} {np.mean(A.mean(1) > 10):10.0%} {drm:>10}")
