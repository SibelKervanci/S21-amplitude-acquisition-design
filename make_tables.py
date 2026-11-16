import config  # ortak yollar ve SMOKE modu
# preds.npz ve stats.json'dan LaTeX tabloları üretir (elle sayı yazılmaz)
import json, numpy as np, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else config.STATS_DIR; OUT = sys.argv[2] if len(sys.argv) > 2 else config.AUTO_DIR
P = np.load(f"{SRC}/preds.npz"); S = json.load(open(f"{SRC}/stats.json"))
NAMES = {"full1": "Full band, 1 thickness", "uni6": "Uniform 6", "dopt6": "D-optimal 6", "uni12": "Uniform 12", "dopt12": "D-optimal 12", "full2": "Full band, 2 thicknesses"}
def metrics(pred, Y, D):
    A = np.abs(pred[:, :12] - Y) / np.abs(Y) * 100
    return np.median(A[:, 0::2]), np.median(A[:, 1::2]), np.mean(A.mean(1) > 10) * 100, np.sqrt(np.mean((pred[:, 12] - D * 1000) ** 2))
def design_table(setn, cfgs, Y, D, label, caption):
    rows = []
    for c in cfgs:
        cells = []
        for m in ("NLLS", "SVR"):
            e, s, g, d = metrics(P[f"{setn}|{m}|{c}"], Y, D); cells.append(f"{e:.1f}/{s:.1f} & {g:.0f} & {d:.1f}")
        rows.append(f"{NAMES[c]} & " + " & ".join(cells) + r" \\")
    t = (r"\begin{table}[!t]" + "\n" + rf"\caption{{{caption}}}" + "\n" + rf"\label{{{label}}}" + "\n" + r"\centering\scriptsize" + "\n" + r"\setlength{\tabcolsep}{2.2pt}" + "\n" +
         r"\begin{tabular}{@{}lcccccc@{}}" + "\n" + r"\toprule" + "\n" + r" & \multicolumn{3}{c}{NLLS} & \multicolumn{3}{c}{SVR}\\" + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n" +
         r"Configuration & $\varepsilon'$/$\sigma_{\mathrm{eff}}$ & $>$10\,\% & $d_1$ & $\varepsilon'$/$\sigma_{\mathrm{eff}}$ & $>$10\,\% & $d_1$\\" + "\n" + r"\midrule" + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" +
         r"\\[2pt]\scriptsize Median APE (\%), gross-error rate (\%), $d_1$ RMSE (mm)." + "\n" + r"\end{table}" + "\n")
    open(f"{OUT}/{label}.tex", "w").write(t)
design_table("syn", ["full1", "uni6", "dopt6", "uni12", "dopt12", "full2"], P["TY"], P["TD"], "tab_design", f"Acquisition Design, $|S_{{21}}|$ Only, Unknown Thickness ($d_1\\sim U(10,40)$~mm, $n={len(P['TD'])}$ Simulated Spectra)")
design_table("ext", ["full1", "uni12", "dopt12", "full2"], P["XY"], P["XD"], "tab_ext", f"External Check on Spectra Computed From 63 Measured Soft-Tissue Records ($n={len(P['XD'])}$ Spectra)")
def sg(x, nd): return f"{x:+.{nd}f}".replace("-", "$-$")
def fmt_p(p): return "$<$0.001" if p < 1e-3 else f"{p:.3f}"
def stats_rows(block):
    out = []
    for t, r in block.items():
        out.append(f"{t} & {sg(r['d_med'],2)} [{sg(r['d_med_ci'][0],2)}, {sg(r['d_med_ci'][1],2)}] & {sg(r['d_gross_pp'],1)} [{sg(r['d_gross_ci'][0],1)}, {sg(r['d_gross_ci'][1],1)}] & {fmt_p(r['p_mcnemar_holm'])} \\\\")
    return out
n_rec = int(np.max(P["ext_groups"])) + 1
t = (r"\begin{table}[!t]" + "\n" + r"\caption{Paired Comparisons (Difference = First Minus Second Method or Configuration)}" + "\n" + r"\label{tab:stats}" + "\n" + r"\centering\scriptsize" + "\n" + r"\setlength{\tabcolsep}{2.5pt}" + "\n" +
     r"\resizebox{\columnwidth}{!}{%" + "\n" + r"\begin{tabular}{@{}lccc@{}}" + "\n" + r"\toprule" + "\n" + r"Comparison & $\Delta$ median APE (\%) & $\Delta$ gross (pp) & $p$ (McNemar, Holm)\\" + "\n" + r"\midrule" + "\n" +
     rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{Simulated spectra ($n={len(P['TD'])}$; spectrum bootstrap)}}}}\\" + "\n" + "\n".join(stats_rows(S["syn"])) + "\n" + r"\midrule" + "\n" +
     rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{Spectra from measured records ({n_rec} records; record-level bootstrap)}}}}\\" + "\n" + "\n".join(stats_rows(S["ext"])) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}}" + "\n" +
     r"\\[2pt]\scriptsize Per-spectrum error = mean APE over the 12 measurands; 95\,\% bootstrap intervals in brackets; positive = first is worse." + "\n" + r"\end{table}" + "\n")
open(f"{OUT}/tab_stats.tex", "w").write(t); print("tablolar yazıldı:", OUT)
