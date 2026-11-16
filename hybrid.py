import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
z = np.load(config.WORK_DIR + '/nothick_data.npz'); Xte, Yte, Dte = z['Xte'], z['Yte'], z['Dte']; sv = np.load(config.WORK_DIR + '/nothick_SVR_log-genlik.npz'); Dml = sv['D']
IDX, OM, e0, c0 = g.TARGET_IDX, g.OMEGA, g.e0, g.c
def mdl(p, d):
    ef, de, tps, a, sg = p; e = ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); zz = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return e, np.abs(((1-G**2)*zz)/(1-G**2*zz**2))
lo, hi = [1,1,3,0,0,0.005], [10,90,20,.5,3.0,0.06]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def fit(obs, d_starts):
    best = None
    for d0 in d_starts:
        for x0 in st5:
            r = least_squares(lambda p: mdl(p[:5], p[5])[1] - obs, x0+[d0], bounds=(lo, hi), x_scale=[1,10,3,.1,.5,.01])
            if best is None or r.cost < best.cost: best = r
    e = mdl(best.x[:5], best.x[5])[0]; return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), best.x[5]
def report(name, P, D, ms):
    A = np.abs(P-Yte)/np.abs(Yte)*100
    print(f"{name:52s} ε′={np.median(A[:,0::2]):4.1f}%  σ_eff={np.median(A[:,1::2]):4.1f}%  >%10 hata={np.mean(A.mean(1)>10):4.0%}  d RMSE={np.sqrt(np.mean((D-Dte)**2))*1000:4.1f} mm  {ms:4.0f} ms")
t = time.time(); out = [fit(Xte[i], [float(np.clip(Dml[i], 0.008, 0.06))]) for i in range(len(Xte))]; ms = (time.time()-t)/len(Xte)*1000
report("HİBRİT: kalınlık başlangıcı = SVR tahmini, 3 başlangıç", np.array([o[0] for o in out]), np.array([o[1] for o in out]), ms)
t = time.time(); out2 = [fit(Xte[i], [0.02, 0.03, 0.04]) for i in range(len(Xte))]; ms2 = (time.time()-t)/len(Xte)*1000
report("NLLS standart: kalınlık 3 başlangıç × 3 = 9 başlangıç", np.array([o[0] for o in out2]), np.array([o[1] for o in out2]), ms2)
