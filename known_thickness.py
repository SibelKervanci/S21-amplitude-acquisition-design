import config  # shared paths and SMOKE mode
# known_thickness.py — |S21|-only retrieval with the slab thickness KNOWN (30 mm): NLLS versus SVR (Section IV-B).
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX = g.OMEGA, g.c, g.e0, g.TARGET_IDX; D0 = 0.030
def s21m(e, d):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/c0)*d); G = (1-n)/(1+n)
    return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def gen(n_per, seed):
    rng = np.random.default_rng(seed); S, Y = [], []
    for tissue, p in g.ITIS_4POLE.items():
        P = g.sample_params(p, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P)
        S.append(g.add_noise(s21m(e, D0), rng, 45, 0.02))
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S), np.vstack(Y)
def eps_model(p):
    ef, de, tps, a, sg = p; return ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
lo, hi = [1, 1, 3, 0, 0], [10, 90, 20, .5, 3.0]; starts = [[4, 45, 8, .1, .5], [3, 30, 6, .05, 1.0], [5, 60, 12, .2, .1]]
def nlls(S):
    best = None
    for x0 in starts:
        r = least_squares(lambda p: s21m(eps_model(p), D0) - S, x0, bounds=(lo, hi), x_scale=[1, 10, 3, .1, .5])
        if best is None or r.cost < best.cost: best = r
    e = eps_model(best.x); return np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel()
def report(name, P, Y, ms=None):
    A = np.abs(P - Y)/np.abs(Y)*100
    print(f"  {name:6s} eps'={np.median(A[:, 0::2]):.1f}%  sigma_eff={np.median(A[:, 1::2]):.1f}%  spectra with mean error >10%: {np.mean(A.mean(1) > 10):.0%}" + (f"   {ms:.0f} ms/spectrum" if ms else ""), flush=True)
T, TY = gen(config.n(12, 3), 7); R, RY = gen(config.n(400, 60), 21)
print(f"Thickness known (30 mm), |S21| only, full band; test n={len(TY)}, SVR training n={len(RY)}; median APE", flush=True)
t = time.time(); report("NLLS", np.array([nlls(s) for s in T]), TY, (time.time()-t)/len(TY)*1000)
sc = StandardScaler().fit(np.log10(np.maximum(R, 1e-6))); ys = StandardScaler().fit(RY)
m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(np.log10(np.maximum(R, 1e-6))), ys.transform(RY))
report("SVR", ys.inverse_transform(m.predict(sc.transform(np.log10(np.maximum(T, 1e-6))))), TY)
