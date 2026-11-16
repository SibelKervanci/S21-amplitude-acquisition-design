import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz
def s21m(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n); return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def eps_model(p):
    ef, de, tps, a, sg = p[:5]; return ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
def gen(n_per, seed):
    rng = np.random.default_rng(seed); S, Y, D = [], [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); d1 = rng.uniform(0.010, 0.040, n_per)
        S.append(g.add_noise(s21m(e, d1[:, None]), rng, 45, 0.02)); D.append(d1)
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S), np.vstack(Y), np.concatenate(D)
lo5, hi5 = [1,1,3,0,0], [10,90,20,.5,3.0]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
# --- doku kütüphanesi: her dokunun nominal 4-kutuplu spektrumuna en iyi uyan etkin 1-kutuplu parametreler ---
lib = []
for t in g.ITIS_4POLE:
    P = g.sample_params(g.ITIS_4POLE[t], 1, np.random.default_rng(0), 0.0, 0.0); obs = s21m(g.eps_4pole(P), 0.03)[0]; b = None
    for x0 in st5:
        r = least_squares(lambda p: s21m(eps_model(p), 0.03) - obs, x0, bounds=(lo5, hi5), x_scale=[1,10,3,.1,.5])
        if b is None or r.cost < b.cost: b = r
    lib.append(b.x)
lib = np.array(lib); LO = np.append(lib.min(0)*0.6, 0.005); HI = np.append(lib.max(0)*1.4, 0.045); LO[3] = 0.0; HI[3] = min(HI[3], 0.5)
# --- D-optimal frekans seçimi (Fisher bilgisi, log|S21|, SNR ağırlıklı) ---
scale = np.array([4, 40, 8, .1, .5, .02])
def rows(p6):
    S0 = s21m(eps_model(p6[:5]), p6[5]); base = np.log(S0); J = np.zeros((111, 6))
    for j in range(6):
        h = 0.005 if j == 3 else 0.01*max(abs(p6[j]), 1e-3); q = p6.copy(); q[j] += h
        J[:, j] = (np.log(s21m(eps_model(q[:5]), q[5])) - base)/h*scale[j]
    slog = np.sqrt((np.sqrt(np.mean(S0**2)/10**4.5)/S0)**2 + 0.02**2); return J/slog[:, None]
Js = [rows(np.append(lib[i], d)) for i in range(len(lib)) for d in (0.015, 0.025, 0.035)]
obj = lambda sel: np.mean([np.linalg.slogdet(J[sel].T@J[sel] + 1e-3*np.eye(6))[1] for J in Js])
sel = []
for k in range(12): sel.append(max([f for f in range(111) if f not in sel], key=lambda f: obj(sel+[f])))
OPT6, OPT12 = np.array(sel[:6]), np.array(sel[:12])
import json as _j; _j.dump({"opt6": [float(x) for x in np.sort(F[OPT6])], "opt12": [float(x) for x in np.sort(F[OPT12])]}, open(config.WORK_DIR + "/dopt_sets.json", "w"))
print("D-optimal seçilen frekanslar (GHz): k=6 ->", np.round(np.sort(F[OPT6]), 2), "| k=12 ->", np.round(np.sort(F[OPT12]), 2), flush=True)
selu = lambda fl: np.array([int(np.argmin(abs(F-x))) for x in fl]); band = lambda a, b: np.where((F >= a-1e-9) & (F <= b+1e-9))[0]
CFG = [("tam bant (111)", np.arange(111)), ("tekdüze 6 frekans", selu([0.5,1,2,3,4,6])), ("D-optimal 6 frekans", OPT6),
       ("tekdüze 12 frekans", selu(np.arange(0.5, 6.01, 0.5))), ("D-optimal 12 frekans", OPT12), ("yüksek bant 4–6", band(4, 6)), ("üst bant 3–6", band(3, 6))]
T, TY, TD = gen(config.n(8, 2), 51); lossy = TY[:, 3] > 0.5; R, RY, RD = gen(config.n(400, 60), 52)
def nlls(S, fi, mode):
    best = None
    if mode == "geniş":   starts, lo, hi = [x+[d] for d in (0.015, 0.025, 0.035) for x in st5], lo5+[0.005], hi5+[0.045]
    else:                 starts, lo, hi = [list(p)+[0.025] for p in lib], list(LO), list(HI)
    for x0 in starts:
        r = least_squares(lambda p: s21m(eps_model(p), p[5])[fi] - S[fi], np.clip(x0, lo, hi), bounds=(lo, hi), x_scale=[1,10,3,.1,.5,.01], max_nfev=80)
        if best is None or r.cost < best.cost: best = r
    e = eps_model(best.x); return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), best.x[5]
def rep(name, P, Dp, ms=None):
    A = np.abs(P-TY)/np.abs(TY)*100; m = A.mean(1) > 10
    print(f"  {name:34s} ε′={np.median(A[:,0::2]):4.1f} σ={np.median(A[:,1::2]):4.1f} | >%10: {np.mean(m):3.0%} | d RMSE {np.sqrt(np.mean((Dp-TD)**2))*1000:4.1f} mm" + (f" | {ms:4.0f} ms" if ms else ""), flush=True)
print(f"\ntest n={len(TD)} (kayıplı %{lossy.mean()*100:.0f}), |S21| tek kalınlık d~U(10,40) mm, SVR eğitim n={len(RD)}")
for mode, title in [("geniş", "NLLS, GENİŞ sınır, 9 başlangıç (önceki)"), ("kütüphane", "NLLS, DOKU KÜTÜPHANESİ başlangıçları + dar sınır (öncül bilgi)")]:
    print(f"\n=== {title} ===", flush=True)
    for name, fi in CFG:
        if mode == "kütüphane" and name.startswith("tam"): pass
        t = time.time(); out = [nlls(T[i], fi, mode) for i in range(len(TD))]; rep(name, np.array([o[0] for o in out]), np.array([o[1] for o in out]), (time.time()-t)/len(TD)*1000)
print("\n=== SVR (log-genlik, öncül = eğitim dağılımı) ===", flush=True)
for name, fi in CFG:
    tgt = np.column_stack([RY, RD*1000]); sc = StandardScaler().fit(np.log10(np.maximum(R[:, fi], 1e-6))); ys = StandardScaler().fit(tgt)
    m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(np.log10(np.maximum(R[:, fi], 1e-6))), ys.transform(tgt))
    o = ys.inverse_transform(m.predict(sc.transform(np.log10(np.maximum(T[:, fi], 1e-6))))); rep(name, o[:, :12], o[:, 12]/1000)
print("BİTTİ", flush=True)
