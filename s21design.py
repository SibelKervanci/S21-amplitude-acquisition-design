import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz; DELTA = 0.010
def s21m(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n)
    return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def gen(n_per, seed):
    rng = np.random.default_rng(seed); S1, S2, Y, D = [], [], [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); d1 = rng.uniform(0.010, 0.040, n_per)
        S1.append(g.add_noise(s21m(e, d1[:, None]), rng, 45, 0.02)); S2.append(g.add_noise(s21m(e, (d1+DELTA)[:, None]), rng, 45, 0.02)); D.append(d1)
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S1), np.vstack(S2), np.vstack(Y), np.concatenate(D)
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
    e = eps_model(best.x); return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), best.x[5]
def summarize(name, P, Y, Dp, Dt, lossy, ms=None):
    A = np.abs(P-Y)/np.abs(Y)*100; m = A.mean(1) > 10
    print(f"  {name:34s} ε′={np.median(A[:,0::2]):4.1f} σ={np.median(A[:,1::2]):4.1f} | alçak f(0.5,1)={np.median(A[:,:4]):4.1f} yüksek f(4,6)={np.median(A[:,8:]):4.1f} | >%10: {np.mean(m):3.0%} (kayıplı {np.mean(m[lossy]):3.0%}, düşük {np.mean(m[~lossy]):3.0%}) | d RMSE {np.sqrt(np.mean((Dp-Dt)**2))*1000:4.1f} mm" + (f" | {ms:4.0f} ms" if ms else ""), flush=True)
def band(a, b): return np.where((F >= a-1e-9) & (F <= b+1e-9))[0]
sel = lambda fl: np.array([int(np.argmin(abs(F-x))) for x in fl])
CFG = [("1 kalınlık | tam bant 0.5–6 (111 nokta)", np.arange(111), False),
       ("1 kalınlık | 6 frekans (0.5,1,2,3,4,6)", sel([0.5,1,2,3,4,6]), False),
       ("1 kalınlık | 12 frekans (0.5 adım)", sel(np.arange(0.5, 6.01, 0.5)), False),
       ("1 kalınlık | alçak bant 0.5–2 GHz", band(0.5, 2.0), False),
       ("1 kalınlık | orta bant 2–4 GHz", band(2.0, 4.0), False),
       ("1 kalınlık | yüksek bant 4–6 GHz", band(4.0, 6.0), False),
       ("1 kalınlık | üst bant 3–6 GHz", band(3.0, 6.0), False),
       ("2 kalınlık (Δ=10mm) | tam bant", np.arange(111), True),
       ("2 kalınlık (Δ=10mm) | 6 frekans", sel([0.5,1,2,3,4,6]), True),
       ("2 kalınlık (Δ=10mm) | yüksek bant 4–6", band(4.0, 6.0), True),
       ("2 kalınlık (Δ=10mm) | üst bant 3–6", band(3.0, 6.0), True)]
T1, T2, TY, TD = gen(config.n(12, 3), 31); lossy = TY[:, 3] > 0.5; R1, R2, RY, RD = gen(config.n(400, 60), 32)
print(f"|S21| genlik, kalınlık bilinmiyor (d1 ~ U(10,40) mm); test n={len(TD)} (kayıplı %{lossy.mean()*100:.0f}); SVR eğitim n={len(RD)}; ε′ ve σ_eff hatası: medyan mutlak yüzde [%]", flush=True)
print("\n=== NLLS (kalınlık serbest, 9 başlangıç) ===", flush=True)
for name, fi, two in CFG:
    t = time.time(); out = [nlls(T1[i], T2[i], fi, two) for i in range(len(TD))]; ms = (time.time()-t)/len(TD)*1000
    summarize(name, np.array([o[0] for o in out]), TY, np.array([o[1] for o in out]), TD, lossy, ms)
print("\n=== SVR (log-genlik, 6800 eğitim) ===", flush=True)
for name, fi, two in CFG:
    tf = lambda a, b: np.log10(np.maximum(np.hstack([a[:, fi]] + ([b[:, fi]] if two else [])), 1e-6))
    tgt = np.column_stack([RY, RD*1000]); sc = StandardScaler().fit(tf(R1, R2)); ys = StandardScaler().fit(tgt)
    m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tf(R1, R2)), ys.transform(tgt))
    o = ys.inverse_transform(m.predict(sc.transform(tf(T1, T2)))); summarize(name, o[:, :12], TY, o[:, 12]/1000, TD, lossy)
print("BİTTİ", flush=True)
