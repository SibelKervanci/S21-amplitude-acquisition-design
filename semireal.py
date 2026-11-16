import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, sys, time, warnings
warnings.filterwarnings('ignore')
pass
from loadreal import *
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
c0, d = 3e8, 0.03
FREQ = np.linspace(0.5, 6.0, 111); W = 2*np.pi*FREQ*1e9
def s21_from_eps(e):
    n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(W/c0)*d); G = (1-n)/(1+n)
    return np.abs(((1-G**2)*z)/(1-G**2*z**2))
def add_noise(S, rng, snr=45, mult=0.02):
    pw = np.mean(S**2, axis=1, keepdims=True)
    Sn = np.clip(S + rng.normal(0, 1, S.shape)*np.sqrt(pw/10**(snr/10)), 1e-6, 1.0)
    return np.clip(Sn*rng.normal(1.0, mult, S.shape), 1e-6, 1.0)
def real_spectra(v):
    lf, lt = np.log(v["f"]), np.log(FREQ*1e9)
    er = np.array([np.interp(lt, lf, v["eps_re"][:, j]) for j in range(v["eps_re"].shape[1])])
    ei = np.array([np.interp(lt, lf, v["eps_im"][:, j]) for j in range(v["eps_im"].shape[1])])
    return s21_from_eps(er - 1j*ei)
D = load_all(config.REAL_DIR)
tr = pd.read_csv(config.TRAIN_CSV)
if config.SMOKE: tr = tr.sample(2500, random_state=0).reset_index(drop=True)
S = [c for c in tr.columns if c.startswith('s21_')]
le = LabelEncoder().fit(tr.tissue); y = le.transform(tr.tissue); X = tr[S].values.astype(np.float32)
a, b = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X[a]); Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
lda = LinearDiscriminantAnalysis(n_components=16).fit(Xa, y[a]); pca = PCA(64, random_state=42).fit(Xa)
t = time.time()
models = {
 "Raw LR":      (LogisticRegression(max_iter=1000).fit(Xa, y[a]), lambda Z: Z),
 "Raw SVM":     (SVC(C=10, gamma='scale').fit(Xa, y[a]), lambda Z: Z),
 "Raw MLP":     (MLPClassifier((256,128), max_iter=300, early_stopping=True, random_state=42).fit(Xa, y[a]), lambda Z: Z),
 "PCA64+RF":    (RandomForestClassifier(300, n_jobs=-1, random_state=42).fit(pca.transform(Xa), y[a]), pca.transform),
 "LDA16+LR":    (LogisticRegression(max_iter=1000).fit(lda.transform(Xa), y[a]), lda.transform),
 "LDA16+SVM":   (SVC(C=10, gamma='scale').fit(lda.transform(Xa), y[a]), lda.transform),
}
print(f"eğitim {time.time()-t:.0f}s")
K = 20; rng = np.random.default_rng(0); real = {}
for rk, cls in MAP.items():
    S0 = real_spectra(D[rk]); real[cls] = (S0, add_noise(np.repeat(S0, K, axis=0), rng))
classes = list(MAP.values())
print("\nSENTETİK hold-out doğruluğu (aynı 7 sınıf, 17 sınıflı tahmin) | GERÇEK-türevli S21 (gürültülü, 17 sınıflı tahmin)")
hdr = f"{'':10s}" + "".join(f"{m:>11s}" for m in models); print(hdr)
def acc_syn(m, f, cls): idx = b[y[b] == le.transform([cls])[0]]; return (models[m][0].predict(models[m][1](sc.transform(X[idx]))) == le.transform([cls])[0]).mean()
def acc_real(m, cls, noisy=True):
    Z = real[cls][1] if noisy else real[cls][0]; p = models[m][0].predict(models[m][1](sc.transform(Z))); return (p == le.transform([cls])[0]).mean()
for tag, fn in [("sentetik", lambda m, c: acc_syn(m, None, c)), ("gerçek+gürültü", lambda m, c: acc_real(m, c, True)), ("gerçek, temiz", lambda m, c: acc_real(m, c, False))]:
    print(f"\n[{tag}]"); print(hdr)
    tot = {m: [] for m in models}; w = {m: 0 for m in models}
    for cls in classes:
        n = real[cls][0].shape[0] if tag != "sentetik" else (y[b] == le.transform([cls])[0]).sum()
        row = f"{cls:10s}"
        for m in models: v = fn(m, cls); row += f"{v:11.2f}"; tot[m].append(v*n); w[m] += n
        print(row + f"   (n={n})")
    print(f"{'AĞIRLIKLI':10s}" + "".join(f"{sum(tot[m])/w[m]:11.3f}" for m in models))
print("\nHeart/Kidney/Spleen gerçek+gürültü: en çok verilen tahminler (Raw MLP):")
for cls in ("Heart","Kidney","Spleen"):
    p = models["Raw MLP"][0].predict(sc.transform(real[cls][1])); vc = pd.Series(le.inverse_transform(p)).value_counts(normalize=True).head(3)
    print(f"  {cls:7s}: " + ", ".join(f"{k} {v:.0%}" for k, v in vc.items()))
