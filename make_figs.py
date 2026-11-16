import config  # ortak yollar ve SMOKE modu
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import generate_dataset_4pole as g
from loadreal import load_all, at_freqs, MAP
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "font.family": "serif", "mathtext.fontset": "stix",
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
F = g.FREQ_GHz; OM = g.OMEGA; e0 = g.e0; OUT = config.FIG_DIR
def eps_nom(t, poles):
    p = g.ITIS_4POLE[t]; e = p["ef"] + 0j
    for k, (d, tt, a, u) in enumerate([(p["d1"], p["t1_ps"], p["a1"], 1e-12), (p["d2"], p["t2_ns"], p["a2"], 1e-9),
                                      (p["d3"], p["t3_us"], p["a3"], 1e-6), (p["d4"], p["t4_ms"], p["a4"], 1e-3)][:poles]):
        if d > 0: e = e + d/(1+(1j*OM*tt*u)**(1-a))
    return e - 1j*p["sig"]/(OM*e0)
# ---- Fig 1: tek kutup vs dört kutup ----
fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True)
for t, c in [("Heart", "#C0392B"), ("Liver", "#2471A3")]:
    e1, e4 = eps_nom(t, 1), eps_nom(t, 4)
    ax[0].plot(F, e4.real, color=c, label=f"{t}, 4-pole"); ax[0].plot(F, e1.real, color=c, ls="--", label=f"{t}, 1-pole")
    ax[1].plot(F, -e4.imag*OM*e0, color=c); ax[1].plot(F, -e1.imag*OM*e0, color=c, ls="--")
ax[0].set_ylabel(r"$\varepsilon'$"); ax[1].set_ylabel(r"$\sigma_{\mathrm{eff}}$ (S/m)"); ax[1].set_xlabel("Frequency (GHz)")
ax[0].legend(ncol=2, frameon=False, loc="upper right"); ax[0].text(0.02, 0.06, "(a)", transform=ax[0].transAxes); ax[1].text(0.02, 0.85, "(b)", transform=ax[1].transAxes)
for a in ax: a.grid(alpha=0.25, lw=0.4)
plt.savefig(f"{OUT}/fig_model_fidelity.pdf"); plt.close()
e1h, e4h = eps_nom("Heart", 1), eps_nom("Heart", 4)
print(f"Heart at 0.5 GHz: sigma_eff = {-e4h.imag[0]*OM[0]*e0:.2f} S/m (four terms) vs {-e1h.imag[0]*OM[0]*e0:.2f} S/m (first dispersion + ionic conductivity); eps difference = {e4h.real[0]-e1h.real[0]:.1f}")
# ---- Fig 2: D-optimal frekanslar ----
OPT12 = [0.5, 0.7, 0.85, 1.0, 1.25, 1.7, 2.35, 2.85, 3.6, 4.4, 5.95, 6.0]; OPT6 = [0.5, 0.85, 1.25, 1.7, 4.4, 6.0]
def s21db(t, d):
    e = eps_nom(t, 4); n = np.sqrt(e); n = np.where(n.imag > 0, np.conj(n), n); z = np.exp(-1j*n*(OM/g.c)*d); G = (1-n)/(1+n)
    return 20*np.log10(np.abs(((1-G**2)*z)/(1-G**2*z**2)))
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for t, c in [("Heart", "#C0392B"), ("Bone_cortical", "#7D3C98"), ("Fat_not_infiltrated", "#B7950B")]:
    ax.plot(F, s21db(t, 0.03), color=c, label=t.replace("_", " ").replace("infiltrated", "infiltr."))
for f in OPT12: ax.axvline(f, color="0.55", lw=0.5, alpha=0.6)
ax.plot(OPT6, [-5]*6, "v", color="k", ms=3.2, label="D-optimal $k{=}6$")
ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel(r"$|S_{21}|$ (dB), $d$=30 mm"); ax.legend(frameon=True, framealpha=1, edgecolor="0.8", loc="lower left", ncol=2, borderpad=0.3); ax.grid(alpha=0.25, lw=0.4)
plt.savefig(f"{OUT}/fig_design_freqs.pdf"); plt.close()
# ---- Fig 4: gerçek doku vs sentetik bulut @1 GHz ----
D = load_all(config.REAL_DIR); syn = pd.read_csv(config.TRAIN_CSV)
cols = {"Heart": "#C0392B", "Kidney": "#2471A3", "Spleen": "#239B56", "Liver": "#B9770E", "Muscle": "#7D3C98", "Brain_white": "#E67EAA"}
real_key = {"Heart": "Heart_standard", "Kidney": "KidneyCortex_standard", "Spleen": "Spleen_standard", "Liver": "Liver_standard", "Muscle": "SkeletalMuscle_standard", "Brain_white": "BrainWhiteMatter_standard"}
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.55))
for a, lst, ttl in [(ax[0], ["Heart", "Kidney", "Spleen", "Liver", "Muscle"], "(a) soft organs"), (ax[1], list(cols) + ["Fat_infiltrated", "Fat_not_infiltrated"], "(b) all mapped tissues, fat")]:
    for t in lst:
        s = syn[syn.tissue == t].sample(250, random_state=0); c = cols.get(t, "0.55")
        a.scatter(s["epsr_f1p0"], s["sigeff_f1p0"], s=2, alpha=0.18, color=c, lw=0)
    for t in lst:
        if t in real_key:
            er, sg = at_freqs(D[real_key[t]], [1.0]); a.scatter(er[:, 0], sg[:, 0], s=14, color=cols[t], edgecolor="k", linewidth=0.5, label=f"{t} (n={len(er)})", zorder=3)
    if a is ax[1]:
        er, sg = at_freqs(D["Fat_standard"], [1.0]); a.scatter(er[:, 0], sg[:, 0], s=14, color="0.55", edgecolor="k", linewidth=0.5, marker="s", label=f"Fat (n={len(er)})", zorder=3)
    a.set_xlabel(r"$\varepsilon'$ at 1 GHz"); a.set_ylabel(r"$\sigma_{\mathrm{eff}}$ at 1 GHz (S/m)"); a.set_title(ttl, fontsize=7, loc="left"); a.grid(alpha=0.25, lw=0.4); a.legend(frameon=False, ncol=2, loc="upper left" if a is ax[0] else "lower right", fontsize=5.6)
plt.savefig(f"{OUT}/fig_real_vs_synth.pdf"); plt.close()
print("ok", os.listdir(OUT))
