import config  # ortak yollar
# generate_gabriel_dataset_v2.py  (EĞER ELİNİZDE ZATEN Dataset_ITIS_ranges_v2.csv VARSA ÇALIŞTIRMAYIN)
# Eski 1 kutuplu veri seti; yalnız domshift.py ("1 kutuplu eğitim -> 4 kutuplu veride çöküş") için gerekir.
# IT'IS V5.0 gerçek parametre aralıklarına göre
import numpy as np
import pandas as pd
import os

np.random.seed(42)

c   = 3e8
e0  = 8.854e-12
D_M = 0.03
FREQ_GHz = np.linspace(0.5, 6.0, 111)

TISSUE_RANGES = {
    "Blood"             : {"eps_inf":(3.5,4.5), "delta_eps1":(45,65),  "tau1_ps":(7.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.5,0.9)},
    "Bone_cancellous"   : {"eps_inf":(2.0,3.0), "delta_eps1":(12,24),  "tau1_ps":(11,16),     "alpha1":(0.15,0.30), "sigma":(0.04,0.10)},
    "Bone_cortical"     : {"eps_inf":(2.0,3.0), "delta_eps1":(7,13),   "tau1_ps":(11,16),     "alpha1":(0.15,0.25), "sigma":(0.01,0.03)},
    "Brain_grey"        : {"eps_inf":(3.5,4.5), "delta_eps1":(35,55),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.01,0.03)},
    "Brain_white"       : {"eps_inf":(3.5,4.5), "delta_eps1":(25,40),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.01,0.04)},
    "Fat_infiltrated"   : {"eps_inf":(2.0,3.0), "delta_eps1":(6,12),   "tau1_ps":(6.5,9.5),   "alpha1":(0.15,0.25), "sigma":(0.02,0.05)},
    "Fat_not_infiltrated":{"eps_inf":(2.0,3.0), "delta_eps1":(1,5),    "tau1_ps":(6.5,9.5),   "alpha1":(0.15,0.25), "sigma":(0.005,0.015)},
    "Heart"             : {"eps_inf":(3.5,4.5), "delta_eps1":(40,60),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.04,0.09)},
    "Kidney"            : {"eps_inf":(3.5,4.5), "delta_eps1":(38,56),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.03,0.08)},
    "Lens_cortex"       : {"eps_inf":(3.5,4.5), "delta_eps1":(33,51),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.20,0.40)},
    "Liver"             : {"eps_inf":(3.5,4.5), "delta_eps1":(30,48),  "tau1_ps":(7.5,10.5),  "alpha1":(0.05,0.15), "sigma":(0.01,0.03)},
    "Lung_inflated"     : {"eps_inf":(2.0,3.0), "delta_eps1":(12,24),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.02,0.05)},
    "Muscle"            : {"eps_inf":(3.5,4.5), "delta_eps1":(40,60),  "tau1_ps":(6.0,8.5),   "alpha1":(0.05,0.15), "sigma":(0.15,0.25)},
    "Skin_dry"          : {"eps_inf":(3.5,4.5), "delta_eps1":(25,40),  "tau1_ps":(6.0,8.5),   "alpha1":(0.00,0.05), "sigma":(0.0001,0.0005)},
    "Skin_wet"          : {"eps_inf":(3.5,4.5), "delta_eps1":(30,48),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.0002,0.0008)},
    "Spleen"            : {"eps_inf":(3.5,4.5), "delta_eps1":(38,58),  "tau1_ps":(6.5,9.5),   "alpha1":(0.05,0.15), "sigma":(0.025,0.06)},
    "Tendon"            : {"eps_inf":(3.5,4.5), "delta_eps1":(33,51),  "tau1_ps":(10,15),     "alpha1":(0.05,0.15), "sigma":(0.18,0.32)},
}
N_SAMPLES = 1000

def cole_cole_gamma(omega, ei, de1, tau1_s, a1, sig):
    eps  = ei + de1 / (1 + (1j * omega * tau1_s)**(1-a1))
    eps -= 1j * sig / (omega * e0)
    return eps

def compute_s21(freq_GHz, eps_c, d=D_M):
    omega = 2 * np.pi * freq_GHz * 1e9
    n = np.sqrt(eps_c.astype(complex))
    n = np.where(np.imag(n) > 0, np.conj(n), n)
    k0 = omega / c
    z  = np.exp(-1j * n * k0 * d)
    G  = (1 - n) / (1 + n)
    return np.abs(((1-G**2)*z) / (1-G**2*z**2))

def awgn(signal, snr_db=45):
    p = np.mean(signal**2)
    noise = np.random.normal(0, np.sqrt(p/(10**(snr_db/10))), len(signal))
    return np.clip(signal + noise, 1e-6, 1.0)

def mixed_noise(signal):
    s = awgn(signal, 45)
    return np.clip(s * np.random.normal(1.0, 0.02, len(s)), 1e-6, 1.0)

omega = 2 * np.pi * FREQ_GHz * 1e9
print("Veri seti üretiliyor...")
rows = []
for tissue, ranges in TISSUE_RANGES.items():
    print(f"  {tissue}: {N_SAMPLES} örnek", flush=True)
    for _ in range(N_SAMPLES):
        ei  = np.random.uniform(*ranges["eps_inf"])
        de1 = np.random.uniform(*ranges["delta_eps1"])
        tau = np.random.uniform(*ranges["tau1_ps"]) * 1e-12
        a1  = np.random.uniform(*ranges["alpha1"])
        sig = np.random.uniform(*ranges["sigma"])
        eps_c     = cole_cole_gamma(omega, ei, de1, tau, a1, sig)
        s21_clean = compute_s21(FREQ_GHz, eps_c, D_M)
        s21_noisy = mixed_noise(s21_clean)
        for i, fg in enumerate(FREQ_GHz):
            rows.append({"tissue": tissue, "f_GHz": round(fg, 6), "S21_Abs": round(s21_noisy[i], 9), "S21_clean": round(s21_clean[i], 9),
                         "eps_inf": round(ei, 4), "delta_eps1": round(de1, 4), "tau1_ps": round(tau*1e12, 6), "alpha1": round(a1, 4),
                         "sigma": round(sig, 6), "snr_db": 45})
df = pd.DataFrame(rows)
outpath = config.ONEPOLE_CSV
os.makedirs(os.path.dirname(outpath), exist_ok=True)
df.to_csv(outpath, index=False)
print(f"\nKaydedildi: {outpath}\nToplam satır: {len(df):,}")
