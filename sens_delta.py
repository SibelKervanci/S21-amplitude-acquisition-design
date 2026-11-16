import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz; DELTA = 0.010
def s21m(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def gen(n_per, seed, a):
    rng = np.random.default_rng(seed); S1, S2, Y, D = [], [], [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); d1 = rng.uniform(0.010, 0.040, n_per); err = rng.uniform(-a, a, n_per)
        S1.append(g.add_noise(s21m(e, d1[:, None]), rng, 45, 0.02)); S2.append(g.add_noise(s21m(e, (d1+DELTA+err)[:, None]), rng, 45, 0.02)); D.append(d1)
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S1), np.vstack(S2), np.vstack(Y), np.concatenate(D)
def eps_model(p):
    ef, de, tps, a, sg = p[:5]; return ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
lo, hi = [1,1,3,0,0,0.005], [10,90,20,.5,3.0,0.045]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def nlls(S1, S2):
    best = None
    for d0 in [0.015, 0.025, 0.035]:
        for x0 in st5:
            f = lambda p: np.concatenate([s21m(eps_model(p), p[5]) - S1, s21m(eps_model(p), p[5]+DELTA) - S2])
            r = least_squares(f, x0+[d0], bounds=(lo, hi), x_scale=[1,10,3,.1,.5,.01])
            if best is None or r.cost < best.cost: best = r
    e = eps_model(best.x); return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), best.x[5]
print("NLLS, 2 kalınlık, tam bant; model Δ=10.0 mm varsayıyor, GERÇEK Δ = 10 ± a mm (düzgün dağılım)")
for a_mm in [0.0, 0.2, 0.5, 1.0, 2.0]:
    S1, S2, Y, D = gen(config.n(12, 3), 41, a_mm/1000); out = [nlls(S1[i], S2[i]) for i in range(len(D))]
    P = np.array([o[0] for o in out]); Dp = np.array([o[1] for o in out]); A = np.abs(P-Y)/np.abs(Y)*100
    print(f"  Δ belirsizliği ±{a_mm:3.1f} mm: ε′={np.median(A[:,0::2]):4.1f}% σ_eff={np.median(A[:,1::2]):4.1f}%  >%10 hata={np.mean(A.mean(1)>10):3.0%}  d1 RMSE={np.sqrt(np.mean((Dp-D)**2))*1000:4.1f} mm", flush=True)
