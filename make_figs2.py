import config  # ortak yollar ve SMOKE modu
import numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
SRC = config.STATS_DIR; OUT = config.FIG_DIR; P = np.load(f"{SRC}/preds.npz")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "font.family": "serif", "mathtext.fontset": "stix", "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
def gross(setn, m, c, Y):
    A = np.abs(P[f"{setn}|{m}|{c}"][:, :12] - Y) / np.abs(Y) * 100; k = int(np.sum(A.mean(1) > 10)); n = len(A)
    ph = k / n; z = 1.96; den = 1 + z**2/n; ctr = (ph + z**2/(2*n))/den; hw = z*np.sqrt(ph*(1-ph)/n + z**2/(4*n**2))/den
    return ph*100, (ctr-hw)*100, (ctr+hw)*100
CF = [("full1", "Full"), ("uni12", "Unif.\n12"), ("dopt12", "D-opt.\n12"), ("uni6", "Unif.\n6"), ("dopt6", "D-opt.\n6"), ("full2", "Full,\n2 thick.")]
fig, ax = plt.subplots(figsize=(3.4, 2.5)); x = np.arange(len(CF)); w = 0.38
for j, (m, col) in enumerate([("NLLS", "#2471A3"), ("SVR", "#CA6F1E")]):
    v = np.array([gross("syn", m, c, P["TY"]) for c, _ in CF]); ax.bar(x + (j - 0.5)*w, v[:, 0], w, yerr=[v[:, 0]-v[:, 1], v[:, 2]-v[:, 0]], capsize=1.5, color=col, label=m, error_kw={"lw": 0.6})
ax.set_xticks(x); ax.set_xticklabels([l for _, l in CF], fontsize=5.8); ax.set_ylabel(r"Gross errors, mean APE $>$10 % (%)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25, lw=0.4)
plt.savefig(f"{OUT}/fig_gross_error.pdf"); plt.close()
# --- grafik özet (IEEE Sensors Journal portalına ayrıca yüklenir): ana mesaj = veri miktarına bağlı geçiş ---
fig = plt.figure(figsize=(7.2, 3.7)); axs = fig.add_axes([0, 0.6, 1, 0.4]); axs.axis("off"); axs.set_xlim(0, 10); axs.set_ylim(0, 3)
def box(x, w, t, c): axs.add_patch(FancyBboxPatch((x, 0.35), w, 2.3, boxstyle="round,pad=0.05", fc=c, ec="0.3", lw=0.8)); axs.text(x + w/2, 1.5, t, ha="center", va="center", fontsize=7)
box(0.1, 2.9, "Simulation study\nLossy slab, thickness unknown, $|S_{21}|$ only\n0.5–6 GHz, four-term Cole–Cole, 17 tissues", "#EAF2F8")
box(3.55, 2.9, "Retrieve $\\varepsilon'$ and $\\sigma_{\\mathrm{eff}}$\nNLLS versus SVR\npaired bootstrap and exact tests", "#FDF2E9")
box(7.0, 2.9, "No method wins everywhere:\nNLLS: dense data, SVR: sparse data\nD-optimal frequencies and a\nsecond thickness help both", "#EAFAF1")
for a, b in [(3.05, 3.5), (6.5, 6.95)]: axs.add_patch(FancyArrowPatch((a, 1.5), (b, 1.5), arrowstyle="-|>", mutation_scale=10, lw=1))
def panel(ax, setn, Y, uni, dopt, title, has6):
    xs = lambda c: {"uni6": 6, "dopt6": 6, "uni12": 12, "dopt12": 12, "full1": 111, "full2": 111}[c]
    for m, col in [("NLLS", "#2471A3"), ("SVR", "#CA6F1E")]:
        u = [(xs(c), gross(setn, m, c, Y)[0]) for c in uni]; d = [(xs(c), gross(setn, m, c, Y)[0]) for c in dopt] + [(111, gross(setn, m, "full1", Y)[0])]
        ax.plot(*zip(*u), "-o", color=col, ms=3.5, lw=1.2, label=f"{m}, uniform"); ax.plot(*zip(*d), "--s", color=col, ms=3.5, lw=1.0, label=f"{m}, D-optimal")
        ax.plot([150], [gross(setn, m, "full2", Y)[0]], marker="*", color=col, ms=8, ls="none", label=f"{m}, full band, 2 thick." if m == "NLLS" else None)
    ax.set_xscale("log"); ax.set_xlim(5, 200); ax.set_xticks([6, 12, 111]); ax.set_xticklabels(["6", "12", "111\n(full)"]); ax.minorticks_off(); ax.set_xlabel("Number of frequencies", fontsize=6.5); ax.set_title(title, fontsize=7); ax.grid(alpha=0.25, lw=0.4)
ax = fig.add_axes([0.07, 0.17, 0.38, 0.36]); panel(ax, "syn", P["TY"], ["uni6", "uni12", "full1"], ["dopt6", "dopt12"], "Simulated spectra ($n=408$)", True); ax.set_ylabel("Gross errors (%)", fontsize=6.5)
ax.legend(frameon=False, fontsize=5.2, loc="upper right")
ax2 = fig.add_axes([0.58, 0.17, 0.38, 0.36]); panel(ax2, "ext", P["XY"], ["uni12", "full1"], ["dopt12"], "From 63 measured records (computed $S_{21}$)", False)
fig.text(0.5, 0.005, "Sparse data (6–12 frequencies): SVR has fewer gross errors.  Full band: NLLS fewer in simulation, no significant difference for measured-record spectra.  Star = full band with two thicknesses.", ha="center", fontsize=5.8, color="0.2")
fig.savefig(f"{OUT}/graphical_abstract.png", dpi=300); fig.savefig(f"{OUT}/graphical_abstract.pdf"); print("ok")
