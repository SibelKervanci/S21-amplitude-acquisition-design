import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings('ignore'); pass
exec(open(config.ROOT + '/semireal.py', encoding='utf-8').read().split("K = 20; rng")[0])
syn = tr
SOFT = ["Heart","Kidney","Spleen","Liver","Muscle","Brain_grey"]; FAT = ["Fat_infiltrated","Fat_not_infiltrated"]
grp = lambda c: "yumuşak organ" if c in SOFT else ("beyaz madde" if c=="Brain_white" else ("yağ" if c in FAT else ("akciğer" if c.startswith("Lung") else ("kemik" if c.startswith("Bone") else "diğer"))))
rng = np.random.default_rng(0)
tests = {"Heart_standard":"yumuşak organ","KidneyCortex_standard":"yumuşak organ","Spleen_standard":"yumuşak organ","Liver_standard":"yumuşak organ","SkeletalMuscle_standard":"yumuşak organ","BrainGreyMatter_standard":"yumuşak organ","BrainWhiteMatter_standard":"beyaz madde","Fat_standard":"yağ","Lung_standard":"akciğer"}
print("Gerçek dokular -> 17 sınıflı sentetik model -> tahminin FİZİKSEL GRUBU (Raw LR | Raw MLP | LDA16+LR), doğru grup oranı")
tot = {m: [0, 0] for m in ("Raw LR","Raw MLP","LDA16+LR")}
for k, g in tests.items():
    S0 = real_spectra(D[k]); Z = add_noise(np.repeat(S0, 20, axis=0), rng); row = f"{k:26s} (n={S0.shape[0]:2d}, beklenen: {g:13s})"
    for m in tot:
        p = le.inverse_transform(models[m][0].predict(models[m][1](sc.transform(Z)))); ok = np.mean([grp(c) == g for c in p]); row += f"  {ok:5.0%}"
        tot[m][0] += ok*S0.shape[0]; tot[m][1] += S0.shape[0]
    print(row)
print(f"{'AĞIRLIKLI (9 gerçek doku grubu)':54s}" + "".join(f"  {tot[m][0]/tot[m][1]:5.0%}" for m in tot))
print("\nReferans: gerçek yağ ve akciğerin 1 GHz'deki medyan ε′ / σ_eff vs sentetik sınıflar")
for k in ("Fat_standard","Lung_standard"):
    er, sg = at_freqs(D[k], [1.0]); print(f"  gerçek {k:14s}: ε′={np.median(er):6.1f}  σ_eff={np.median(sg):.3f}")
for c in ("Fat_infiltrated","Fat_not_infiltrated","Lung_inflated"):
    s = syn[syn.tissue == c]; print(f"  sentetik {c:20s}: ε′={s['epsr_f1p0'].median():6.1f}  σ_eff={s['sigeff_f1p0'].median():.3f}")
