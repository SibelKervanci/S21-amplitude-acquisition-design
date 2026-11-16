import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz; u = (F-F[0])/(F[-1]-F[0]); D0 = 0.030
def s21c(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d[:, None]); G = (1-n)/(1+n); return ((1-G**2)*z)/(1-G**2*z**2)
def gen(n_per, seed, nonideal):
    rng = np.random.default_rng(seed); Xs, Ys = [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); N = n_per
        if nonideal:
            d = D0 + rng.uniform(-0.001, 0.001, N)                                   # kalınlık ±1 mm (bilinmeyen sapma)
            S = s21c(e, d)
            a0, a1, a2 = rng.uniform(-.3, .3, N), rng.uniform(-.3, .3, N), rng.uniform(0, .2, N); k = rng.uniform(2, 4, N); ph = rng.uniform(0, 2*np.pi, N)
            gdb = a0[:, None] + a1[:, None]*u[None, :] + a2[:, None]*np.sin(2*np.pi*k[:, None]*u[None, :] + ph[:, None]); S = S*10**(gdb/20)   # iletim izleme hatası (dB): offset+eğim+dalgalanma
            xt = 10**(rng.uniform(-60, -45, N)/20); S = S + (xt*np.exp(1j*rng.uniform(0, 2*np.pi, N)))[:, None]                                # sızıntı/crosstalk tabanı
            Sm = np.abs(S)
        else:
            Sm = np.abs(s21c(e, np.full(N, D0)))
        Xs.append(g.add_noise(Sm, rng, 45, 0.02)); Ys.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(Xs), np.vstack(Ys)
tf = lambda X: np.log10(np.maximum(X, 1e-6))
Xi, Yi = gen(config.n(1000, 100), 1, False); Xn, Yn = gen(config.n(1000, 100), 2, True)
rs = np.random.default_rng(0); si, sn = rs.choice(len(Xi), min(8000, len(Xi)), replace=False), rs.choice(len(Xn), min(8000, len(Xn)), replace=False)
Ti = gen(config.n(18, 3), 3, False); Tn = gen(config.n(18, 3), 4, True)
print(f"hazır; ideal medyan |S21| 6GHz={np.median(Xi[:,110]):.4f}  kusurlu={np.median(Xn[:,110]):.4f}", flush=True)
def fit_svr(X, Y):
    sc = StandardScaler().fit(tf(X)); ys = StandardScaler().fit(Y); m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tf(X)), ys.transform(Y))
    return lambda Z: ys.inverse_transform(m.predict(sc.transform(tf(Z))))
t = time.time(); svr_naive = fit_svr(Xi[si], Yi[si]); print(f"SVR naif (ideal veriyle) {time.time()-t:.0f}s", flush=True)
t = time.time(); svr_dr = fit_svr(Xn[sn], Yn[sn]); print(f"SVR rastgeleleştirilmiş (kusurlu veriyle) {time.time()-t:.0f}s", flush=True)
def mdl(p, d):
    ef, de, tps, a, sg = p[:5]; e = ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return e, np.abs(((1-G**2)*z)/(1-G**2*z**2))
lo5, hi5 = [1,1,3,0,0], [10,90,20,.5,3.0]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def nlls(obs, free_d):
    best = None
    for x0 in st5:
        if free_d: f = lambda p: mdl(p, p[5])[1] - obs; lo, hi, x, xs = lo5+[0.02], hi5+[0.04], x0+[0.03], [1,10,3,.1,.5,.003]
        else:      f = lambda p: mdl(p, D0)[1] - obs;   lo, hi, x, xs = lo5, hi5, x0, [1,10,3,.1,.5]
        r = least_squares(f, x, bounds=(lo, hi), x_scale=xs)
        if best is None or r.cost < best.cost: best = r
    e = mdl(best.x, best.x[5] if free_d else D0)[0]; return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel()
def rep(name, P, Y, ms=None):
    A = np.abs(P-Y)/np.abs(Y)*100; print(f"  {name:44s} ε′={np.median(A[:,0::2]):5.1f}%  σ_eff={np.median(A[:,1::2]):5.1f}%  >%10 hata={np.mean(A.mean(1)>10):4.0%}" + (f"  {ms:4.0f} ms" if ms else ""), flush=True)
for title, (X, Y) in [("TEST 1: İDEAL ölçüm (modelle tam uyumlu)", Ti), ("TEST 2: KUSURLU ölçüm (izleme hatası + sızıntı + kalınlık ±1 mm; NLLS'in bilmediği)", Tn)]:
    print("\n" + title); rep("SVR naif (ideal veriyle eğitilmiş)", svr_naive(X), Y); rep("SVR rastgeleleştirilmiş (kusurlu veriyle eğitilmiş)", svr_dr(X), Y)
    for nm, fd in [("NLLS, kalınlık 30 mm sabit", False), ("NLLS, kalınlık serbest", True)]:
        t = time.time(); P = np.array([nlls(X[i], fd) for i in range(len(X))]); rep(nm, P, Y, (time.time()-t)/len(X)*1000)
print("BİTTİ", flush=True)
