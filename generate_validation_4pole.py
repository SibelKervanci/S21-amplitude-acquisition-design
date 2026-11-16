import config  # ortak yollar ve SMOKE modu
# generate_validation_4pole.py
# ═══════════════════════════════════════════════════════════════════
# IT'IS V5.0 Harici Validasyon Veri Seti — 4 Kutuplu Cole-Cole
#
# Eğitim seti (Dataset_ITIS_ranges_v2.csv):
#   → IT'IS min-max aralıklarından uniform random, 1 kutup
#
# Bu validasyon seti:
#   → IT'IS nominal değerleri ± %20 varyasyon, 4 kutup
#   → Fiziksel olarak daha gerçekçi (tam dispersiyon spektrumu)
#   → Eğitim dağılımından farklı → gerçek domain shift testi
#
# Makale ifadesi:
#   "External validation was performed on a secondary synthetic
#    dataset generated using the full four-pole Cole-Cole model
#    with IT'IS V5.0 nominal parameters ± 20% variability,
#    representing a physically more realistic distribution shift."
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import os

np.random.seed(123)  # Eğitim setinden farklı seed!

c   = 3e8
e0  = 8.854e-12
D_M = 0.03
FREQ_GHz = np.linspace(0.5, 6.0, 111)
omega    = 2 * np.pi * FREQ_GHz * 1e9

# ── IT'IS V5.0 Nominal Değerleri — 4 Kutuplu Cole-Cole ───────────
# Kaynak: IT'IS Foundation V5.0, DOI: 10.13099/VIP21000-05-0
# Kolonlar: ef, d1, t1(ps), a1, d2, t2(ns), a2, sig,
#            d3, t3(µs), a3, d4, t4(ms), a4
ITIS_4POLE = {
    "Blood": {
        "ef":4.0,  "d1":56.0,  "t1_ps":8.377,  "a1":0.10,
        "d2":5200, "t2_ns":132.629, "a2":0.10,  "sig":0.700,
        "d3":0,    "t3_us":159.155, "a3":0.20,
        "d4":0,    "t4_ms":15.915,  "a4":0.00
    },
    "Bone_cancellous": {
        "ef":2.5,  "d1":18.0,  "t1_ps":13.263, "a1":0.22,
        "d2":300,  "t2_ns":79.577,  "a2":0.25,  "sig":0.070,
        "d3":20000,"t3_us":159.155, "a3":0.20,
        "d4":20000000, "t4_ms":15.915, "a4":0.00
    },
    "Bone_cortical": {
        "ef":2.5,  "d1":10.0,  "t1_ps":13.263, "a1":0.20,
        "d2":180,  "t2_ns":79.577,  "a2":0.20,  "sig":0.020,
        "d3":5000, "t3_us":159.155, "a3":0.20,
        "d4":100000, "t4_ms":15.915, "a4":0.00
    },
    "Brain_grey": {
        "ef":4.0,  "d1":45.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":400,  "t2_ns":15.915,  "a2":0.15,  "sig":0.020,
        "d3":200000, "t3_us":106.103, "a3":0.22,
        "d4":45000000, "t4_ms":5.305, "a4":0.00
    },
    "Brain_white": {
        "ef":4.0,  "d1":32.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":100,  "t2_ns":7.958,   "a2":0.10,  "sig":0.020,
        "d3":40000,"t3_us":53.052,  "a3":0.30,
        "d4":35000000, "t4_ms":7.958, "a4":0.02
    },
    "Fat_infiltrated": {
        "ef":2.5,  "d1":9.0,   "t1_ps":7.958,  "a1":0.20,
        "d2":35,   "t2_ns":15.915,  "a2":0.10,  "sig":0.035,
        "d3":33000,"t3_us":159.155, "a3":0.05,
        "d4":10000000, "t4_ms":15.915, "a4":0.01
    },
    "Fat_not_infiltrated": {
        "ef":2.5,  "d1":3.0,   "t1_ps":7.958,  "a1":0.20,
        "d2":15,   "t2_ns":15.915,  "a2":0.10,  "sig":0.010,
        "d3":33000,"t3_us":159.155, "a3":0.05,
        "d4":10000000, "t4_ms":7.958, "a4":0.01
    },
    "Heart": {
        "ef":4.0,  "d1":50.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":1200, "t2_ns":159.155, "a2":0.05,  "sig":0.050,
        "d3":450000, "t3_us":72.343, "a3":0.22,
        "d4":25000000, "t4_ms":4.547, "a4":0.00
    },
    "Kidney": {
        "ef":4.0,  "d1":47.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":3500, "t2_ns":198.944, "a2":0.22,  "sig":0.050,
        "d3":250000, "t3_us":79.577, "a3":0.22,
        "d4":30000000, "t4_ms":4.547, "a4":0.00
    },
    "Lens_cortex": {
        "ef":4.0,  "d1":42.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":1500, "t2_ns":79.577,  "a2":0.10,  "sig":0.300,
        "d3":200000, "t3_us":159.155, "a3":0.10,
        "d4":40000000, "t4_ms":15.915, "a4":0.00
    },
    "Liver": {
        "ef":4.0,  "d1":39.0,  "t1_ps":8.842,  "a1":0.10,
        "d2":6000, "t2_ns":530.516, "a2":0.20,  "sig":0.020,
        "d3":50000,"t3_us":22.736,  "a3":0.20,
        "d4":30000000, "t4_ms":15.915, "a4":0.05
    },
    "Lung_inflated": {
        "ef":2.5,  "d1":18.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":500,  "t2_ns":63.662,  "a2":0.10,  "sig":0.030,
        "d3":250000, "t3_us":159.155, "a3":0.20,
        "d4":40000000, "t4_ms":7.958, "a4":0.00
    },
    "Muscle": {
        "ef":4.0,  "d1":50.0,  "t1_ps":7.234,  "a1":0.10,
        "d2":7000, "t2_ns":353.678, "a2":0.10,  "sig":0.200,
        "d3":1200000, "t3_us":318.310, "a3":0.10,
        "d4":25000000, "t4_ms":2.274, "a4":0.00
    },
    "Skin_dry": {
        "ef":4.0,  "d1":32.0,  "t1_ps":7.234,  "a1":0.00,
        "d2":1100, "t2_ns":32.481,  "a2":0.20,  "sig":0.0002,
        "d3":0,    "t3_us":159.155, "a3":0.20,
        "d4":0,    "t4_ms":15.915,  "a4":0.20
    },
    "Skin_wet": {
        # IT'IS'te ayrı "Skin Wet" yok — Skin + ıslak düzeltme
        "ef":4.0,  "d1":39.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":280,  "t2_ns":79.577,  "a2":0.16,  "sig":0.0004,
        "d3":30000,"t3_us":159.155, "a3":0.20,
        "d4":30000,"t4_ms":1.592,   "a4":0.20
    },
    "Spleen": {
        "ef":4.0,  "d1":48.0,  "t1_ps":7.958,  "a1":0.10,
        "d2":2500, "t2_ns":63.662,  "a2":0.15,  "sig":0.030,
        "d3":200000, "t3_us":265.258, "a3":0.25,
        "d4":50000000, "t4_ms":6.366, "a4":0.00
    },
    "Tendon": {
        "ef":4.0,  "d1":42.0,  "t1_ps":12.243, "a1":0.10,
        "d2":60,   "t2_ns":6.366,   "a2":0.10,  "sig":0.250,
        "d3":60000,"t3_us":318.310, "a3":0.22,
        "d4":20000000, "t4_ms":1.326, "a4":0.00
    },
}

N_SAMPLES = 200   # her doku için — validasyon seti daha küçük
VAR       = 0.20  # ±%20 varyasyon


# ── 4 Kutuplu Cole-Cole ──────────────────────────────────────────
def cole_cole_4pole(omega, ef, d1,t1,a1, d2,t2,a2,
                              d3,t3,a3, d4,t4,a4, sig):
    """
    4 kutuplu Cole-Cole dispersiyon modeli.
    Tüm zaman sabitleri SI biriminde (saniye).
    """
    eps = ef + 0j
    # Kutup 1: GHz (~ps)
    if d1 > 0 and t1 > 0:
        eps += d1 / (1 + (1j * omega * t1)**(1 - a1))
    # Kutup 2: MHz (~ns)
    if d2 > 0 and t2 > 0:
        eps += d2 / (1 + (1j * omega * t2)**(1 - a2))
    # Kutup 3: kHz (~µs)
    if d3 > 0 and t3 > 0:
        eps += d3 / (1 + (1j * omega * t3)**(1 - a3))
    # Kutup 4: Hz (~ms)
    if d4 > 0 and t4 > 0:
        eps += d4 / (1 + (1j * omega * t4)**(1 - a4))
    # İletkenlik
    eps -= 1j * sig / (omega * e0)
    return eps


def compute_s21(freq_GHz, eps_c, d=D_M):
    omega2 = 2 * np.pi * freq_GHz * 1e9
    n = np.sqrt(eps_c.astype(complex))
    n = np.where(np.imag(n) > 0, np.conj(n), n)
    k0 = omega2 / c
    z  = np.exp(-1j * n * k0 * d)
    G  = (1 - n) / (1 + n)
    return np.abs(((1-G**2)*z) / (1-G**2*z**2))


def awgn(signal, snr_db=45):
    p = np.mean(signal**2)
    noise = np.random.normal(0,
                np.sqrt(p/(10**(snr_db/10))), len(signal))
    return np.clip(signal + noise, 1e-6, 1.0)


def mixed_noise(signal):
    s = awgn(signal, 45)
    return np.clip(
        s * np.random.normal(1.0, 0.02, len(s)), 1e-6, 1.0)


def vary(val, pct=VAR):
    """Nominal değere ±pct varyasyon ekle."""
    if val == 0:
        return 0.0
    low  = val * (1 - pct)
    high = val * (1 + pct)
    return max(np.random.uniform(low, high), 1e-10)


# ── Veri üretimi ─────────────────────────────────────────────────
print("=" * 60)
print("4-Pole Cole-Cole Validation Dataset Generation")
print(f"Source: IT'IS Foundation V5.0 ± {VAR*100:.0f}% variability")
print(f"Samples: {N_SAMPLES}/tissue × {len(ITIS_4POLE)} tissues")
print("=" * 60)

rows = []
for tissue, p in ITIS_4POLE.items():
    print(f"  {tissue}: {N_SAMPLES} samples...", flush=True)
    for _ in range(N_SAMPLES):
        # Tüm parametrelere varyasyon ekle
        ef_v  = vary(p["ef"],    0.10)   # ε∞ ±%10
        d1_v  = vary(p["d1"],    VAR)
        t1_v  = vary(p["t1_ps"], VAR) * 1e-12  # ps → s
        a1_v  = min(vary(p["a1"], 0.30), 0.5) if p["a1"] > 0 \
                else np.random.uniform(0, 0.05)
        d2_v  = vary(p["d2"],    VAR) if p["d2"] > 0 else 0
        t2_v  = vary(p["t2_ns"], VAR) * 1e-9   # ns → s
        a2_v  = min(vary(p["a2"], 0.30), 0.5)  if p["a2"] > 0 else 0
        d3_v  = vary(p["d3"],    VAR) if p["d3"] > 0 else 0
        t3_v  = vary(p["t3_us"], VAR) * 1e-6   # µs → s
        a3_v  = min(vary(p["a3"], 0.30), 0.5)  if p["a3"] > 0 else 0
        d4_v  = vary(p["d4"],    VAR) if p["d4"] > 0 else 0
        t4_v  = vary(p["t4_ms"], VAR) * 1e-3   # ms → s
        a4_v  = min(vary(p["a4"], 0.30), 0.5)  if p["a4"] > 0 else 0
        sig_v = vary(p["sig"],   VAR)

        # 4 kutuplu permittivite hesapla
        eps_c = cole_cole_4pole(
            omega,
            ef_v,
            d1_v, t1_v, a1_v,
            d2_v, t2_v, a2_v,
            d3_v, t3_v, a3_v,
            d4_v, t4_v, a4_v,
            sig_v
        )

        s21_clean = compute_s21(FREQ_GHz, eps_c, D_M)
        s21_noisy = mixed_noise(s21_clean)

        for i, fg in enumerate(FREQ_GHz):
            rows.append({
                "tissue"    : tissue,
                "f_GHz"     : round(fg, 6),
                "S21_Abs"   : round(s21_noisy[i], 9),
                "S21_clean" : round(s21_clean[i], 9),
                # 1. kutup parametreleri — model tahmini için
                "eps_inf"   : round(ef_v, 4),
                "delta_eps1": round(d1_v, 4),
                "tau1_ps"   : round(t1_v * 1e12, 6),
                "alpha1"    : round(a1_v, 4),
                "sigma"     : round(sig_v, 6),
                # 4 kutup bilgisi — referans için
                "d2"        : round(d2_v, 4),
                "tau2_ns"   : round(t2_v * 1e9, 6),
                "alpha2"    : round(a2_v, 4),
                "d3"        : round(d3_v, 4),
                "tau3_us"   : round(t3_v * 1e6, 6),
                "alpha3"    : round(a3_v, 4),
                "d4"        : round(d4_v, 4),
                "tau4_ms"   : round(t4_v * 1e3, 6),
                "alpha4"    : round(a4_v, 4),
                "snr_db"    : 45,
                "n_poles"   : 4,
            })

df = pd.DataFrame(rows)
outpath = config.VAL4_CSV
df.to_csv(outpath, index=False)

print(f"\nKaydedildi: {outpath}")
print(f"Toplam satır: {len(df):,}")
print(f"Her doku: {N_SAMPLES} örnek × {len(FREQ_GHz)} frekans "
      f"= {N_SAMPLES*len(FREQ_GHz):,} satır")
print(f"Toplam örnek: {len(ITIS_4POLE)*N_SAMPLES:,}")

# Özet istatistik
print("\nDataset Summary:")
print(f"  Tissues     : {df['tissue'].nunique()}")
print(f"  Frequencies : {df['f_GHz'].nunique()}")
print(f"  Total rows  : {len(df):,}")
print(f"  Model       : 4-pole Cole-Cole")
print(f"  Variability : ±{VAR*100:.0f}%")
print(f"  Noise       : AWGN 45dB + 2% multiplicative")
print("\nTissue sample counts:")
print(df.groupby('tissue')['f_GHz'].count() // len(FREQ_GHz))
