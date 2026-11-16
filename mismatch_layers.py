import config  # ortak yollar ve SMOKE modu
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import generate_dataset_4pole as g
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
OM, c0, e0, IDX, F = g.OMEGA, g.c, g.e0, g.TARGET_IDX, g.FREQ_GHz; K0 = OM/c0
def slab(e, d):                                   # tek levha (ters çözümün varsaydığı model)
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*K0*d); G = (1-n)/(1+n)
    return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def layered(layers, sign):                        # çok katmanlı iletim (transfer matrisi), hava | katmanlar | hava
    m11 = np.ones_like(layers[0][0], dtype=complex); m22 = m11.copy(); m12 = np.zeros_like(m11); m21 = np.zeros_like(m11)
    for n, d in layers:
        n = np.where(n.imag > 0, np.conj(n), n); dl = K0*n*d; c, s = np.cos(dl), np.sin(dl)
        a11, a12, a21, a22 = c, sign*1j*s/n, sign*1j*n*s, c
        m11, m12, m21, m22 = m11*a11 + m12*a21, m11*a12 + m12*a22, m21*a11 + m22*a21, m21*a12 + m22*a22
    return np.abs(2/(m11 + m12 + m21 + m22))
# işaret/birim kontrolü: katmansız (yalnız numune) durum, kapalı formülle aynı olmalı
p = g.ITIS_4POLE["Muscle"]; e_t = g.eps_4pole(g.sample_params(p, 1, np.random.default_rng(0), 0.1, 0.02)); dd = np.array([[0.02]])
ref = slab(e_t, dd); best = None
for sg in (+1, -1):
    err = np.max(np.abs(layered([(np.sqrt(e_t), dd)], sg) - ref)); best = (err, sg) if best is None or err < best[0] else best
print(f"transfer-matris doğrulaması: en iyi işaret {best[1]:+d}, maks. fark {best[0]:.2e}", flush=True); SG = best[1]; assert best[0] < 1e-9
def gen(n_per, seed, multilayer):
    rng = np.random.default_rng(seed); S, Y, D = [], [], []
    for tissue, pp in g.ITIS_4POLE.items():
        P = g.sample_params(pp, n_per, rng, 0.15, 0.03); e = g.eps_4pole(P); d1 = rng.uniform(0.010, 0.040, n_per)[:, None]
        if multilayer:
            nw = np.full(e.shape, np.sqrt(2.05*(1 - 1j*2e-4)), dtype=complex); one = np.ones(e.shape, dtype=complex)
            gap1, gap2 = rng.uniform(0, 0.0005, (n_per, 1)), rng.uniform(0, 0.0005, (n_per, 1)); w = np.full((n_per, 1), 0.001)   # 1 mm PTFE pencere, 0-0.5 mm hava aralığı
            M = layered([(nw, w), (one, gap1), (np.sqrt(e), d1), (one, gap2), (nw, w)], SG)
        else: M = slab(e, d1)
        S.append(g.add_noise(M, rng, 45, 0.02)); D.append(d1[:, 0])
        Y.append(np.column_stack([np.column_stack([e[:, i].real, -e[:, i].imag*OM[i]*e0]) for i in IDX]))
    return np.vstack(S), np.vstack(Y), np.concatenate(D)
Tx, Ty, Td = gen(config.n(12, 3), 61, True); Ti, Ty_i, Td_i = gen(config.n(12, 3), 62, False)
Rx_i, Ry_i, Rd_i = gen(config.n(400, 60), 63, False); Rx_m, Ry_m, Rd_m = gen(config.n(400, 60), 64, True)
tf = lambda X: np.log10(np.maximum(X, 1e-6))
def fit_svr(X, Y, D):
    t = np.column_stack([Y, D*1000]); sc = StandardScaler().fit(tf(X)); ys = StandardScaler().fit(t)
    m = MultiOutputRegressor(SVR(C=10, epsilon=0.05, gamma='scale'), n_jobs=-1).fit(sc.transform(tf(X)), ys.transform(t))
    return lambda Z: ys.inverse_transform(m.predict(sc.transform(tf(Z))))
def eps_model(p):
    ef, de, tps, a, sg = p[:5]; return ef + de/(1+(1j*OM*tps*1e-12)**(1-a)) - 1j*sg/(OM*e0)
lo, hi = [1,1,3,0,0,0.005], [10,90,20,.5,3.0,0.045]; st5 = [[4,45,8,.1,.5],[3,30,6,.05,1.0],[5,60,12,.2,.1]]
def nlls(S):
    best = None
    for d0 in (0.015, 0.025, 0.035):
        for x0 in st5:
            r = least_squares(lambda p: slab(eps_model(p), p[5]) - S, x0+[d0], bounds=(lo, hi), x_scale=[1,10,3,.1,.5,.01])
            if best is None or r.cost < best.cost: best = r
    e = eps_model(best.x); return np.concatenate([np.column_stack([e.real[IDX], -e.imag[IDX]*OM[IDX]*e0]).ravel(), [best.x[5]*1000]])
def rep(name, out, Y, D):
    P, dp = out[:, :12], out[:, 12]/1000; A = np.abs(P-Y)/np.abs(Y)*100
    print(f"  {name:46s} ε′={np.median(A[:,0::2]):4.1f} σ={np.median(A[:,1::2]):4.1f} | >%10: {np.mean(A.mean(1)>10):3.0%} | d RMSE {np.sqrt(np.mean((dp-D)**2))*1000:4.1f} mm", flush=True)
svr_i = fit_svr(Rx_i, Ry_i, Rd_i); print("SVR (tek levha verisiyle) hazır", flush=True)
svr_m = fit_svr(Rx_m, Ry_m, Rd_m); print("SVR (pencere+aralıklı veriyle) hazır", flush=True)
for title, X, Y, D in [("TEST A: veri = tek levha formülü (ters suç VAR)", Ti, Ty_i, Td_i), ("TEST B: veri = pencere + hava aralığı içeren çok katmanlı model (ters çözümün bilmediği; ters suç YOK)", Tx, Ty, Td)]:
    print("\n" + title, flush=True); rep("SVR, tek levha verisiyle eğitilmiş", svr_i(X), Y, D); rep("SVR, çok katmanlı verisiyle eğitilmiş", svr_m(X), Y, D)
    t = time.time(); out = np.array([nlls(X[i]) for i in range(len(D))]); rep(f"NLLS, tek levha modeli (kalınlık serbest) [{(time.time()-t)/len(D)*1000:.0f} ms]", out, Y, D)
print("BİTTİ", flush=True)
