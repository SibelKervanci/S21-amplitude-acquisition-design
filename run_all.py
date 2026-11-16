# run_all.py — betikleri doğru sırayla çalıştırır, her birinin çıktısını outputs/logs/ altına kaydeder.
#   python run_all.py --list                 adımları listele
#   python run_all.py --smoke                çok küçük boyutlarla hızlı deneme (birkaç dakika; sonuçlar makaleye girmez)
#   python run_all.py                        tam çalıştırma (toplam yaklaşık 2 saat, tek çekirdek)
#   python run_all.py --only data stats mlp tables      yalnız seçilen adımlar
#   python run_all.py --skip domshift latex             bazı adımları atla
#   python run_all.py --keep-going           bir adım hata verse de devam et
import os, sys, time, subprocess, argparse, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [  # (ad, betikler, açıklama, yaklaşık dk)
 ("data",     ["generate_dataset_4pole.py"], "Sentetik 4 kutuplu eğitim verisi (gabriel_datasets/Dataset_4pole_train.csv)", 1),
 ("fig_basic",["make_figs.py"], "Şekil 1, 2, 4 (real_data/ klasöründeki .ods dosyaları gerekir)", 1),
 ("dopt",     ["optdesign.py"], "D-optimal frekans kümeleri + kütüphane başlangıçlı NLLS (n=136)", 20),
 ("stats",    ["stats_design.py"], "ANA deney: NLLS ve SVR, 408 sim. + 252 gerçek-türevli spektrum, istatistikler", 20),
 ("mlp",      ["stats_mlp.py"], "MLP-SVR karşılaştırması (Bölüm V-D)", 3),
 ("tables",   ["make_tables.py", "make_figs2.py"], "Tablo III, IV, VII (paper/auto/*.tex), Şekil 3, grafik özet", 1),
 ("thick",    ["nothick.py", "hybrid.py"], "Tablo II: bilinmeyen kalınlık 10-50 mm + hibrit NLLS", 15),
 ("imperf",   ["nonideal.py"], "Tablo V: kusurlu ölçüm", 6),
 ("forward",  ["mismatch_layers.py"], "Farklı ileri model (PTFE pencere + hava aralığı)", 4),
 ("delta",    ["sens_delta.py"], "Kalınlık adımı (Delta) hassasiyeti", 4),
 ("bands",    ["s21design.py"], "Bant seçimi (0.5-2, 2-4, 4-6 GHz) ve iki kalınlık (n=204)", 25),
 ("known",    ["known_thickness.py"], "Bilinen kalınlıkta (30 mm) NLLS ve SVR, yalnız |S21| (n=204)", 3),
 ("real",     ["envelope_check.py", "semireal.py", "reg_real.py", "nlls_real.py", "diag2.py", "diag3.py"], "Gerçek-türevli sınıflandırma, retrieval, ayrılabilirlik", 15),
 ("domshift", ["generate_validation_4pole.py", "domshift.py"], "(İSTEĞE BAĞLI) 1 kutuplu -> 4 kutuplu çöküş; eski 1 kutuplu CSV gerekir", 10),
 ("latex",    [], "paper/ klasöründe pdflatex + bibtex (kuruluysa)", 1),
]
def need_check():
    miss = []
    for m, pip in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("odf", "odfpy")]:
        try: __import__(m)
        except Exception: miss.append(pip)
    return miss
def run_script(script, env, log_dir):
    log = os.path.join(log_dir, script.replace(".py", ".log")); t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([sys.executable, "-u", script], cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail = []
        for raw in p.stdout:
            line = raw.decode("utf-8", errors="replace"); lf.write(line); lf.flush(); tail.append(line.rstrip()); tail = tail[-3:]
        rc = p.wait()
    dt = time.time() - t0
    print(f"   {'OK ' if rc == 0 else 'HATA'} {script:30s} {dt/60:6.1f} dk   (log: {os.path.relpath(log, HERE)})")
    if rc != 0: print("      son satırlar:\n      " + "\n      ".join(tail))
    return rc
def run_latex():
    pd_ = os.path.join(HERE, "paper")
    if not shutil.which("pdflatex"): print("   pdflatex bulunamadı: paper/ klasörünü Overleaf'e yükleyin (main.tex)."); return 0
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        r = subprocess.run(cmd, cwd=pd_, capture_output=True); 
        if r.returncode != 0 and cmd[0] == "pdflatex": print("   LaTeX uyarı/hata (main.log'a bakın)")
    print("   paper/main.pdf üretildi" if os.path.exists(os.path.join(pd_, "main.pdf")) else "   main.pdf üretilemedi"); return 0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--keep-going", action="store_true"); a = ap.parse_args()
    if a.list:
        tot = 0
        for i, (n, sc, d, m) in enumerate(STEPS, 1): print(f"{i:2d}. {n:10s} ~{m:3d} dk  {', '.join(sc) or '-':45s} {d}"); tot += m
        print(f"    toplam ~{tot} dk"); return
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"; env["REPRO_DIR"] = HERE; env["SMOKE"] = "1" if a.smoke else "0"
    sys.path.insert(0, HERE); os.environ["SMOKE"] = env["SMOKE"]; import config
    miss = need_check()
    if miss: print("EKSİK PAKET:", ", ".join(miss), "\n  -> pip install " + " ".join(miss)); return
    nods = [f for f in os.listdir(config.REAL_DIR)] if os.path.isdir(config.REAL_DIR) else []
    if not any(f.endswith(".ods") for f in nods): print(f"UYARI: {config.REAL_DIR} içinde .ods yok; 'fig_basic', 'stats', 'real' adımları çalışmaz (CORA.RDR dosyalarını oraya koyun).")
    print(f"Mod: {'SMOKE (hızlı deneme)' if a.smoke else 'TAM'} | çıktı klasörü: {os.path.relpath(config.OUT_ROOT, HERE)}\n"); T0 = time.time(); fails = []
    for n, scripts, desc, m in STEPS:
        if (a.only and n not in a.only) or n in a.skip: continue
        print(f"[{n}] {desc}  (~{m} dk)")
        if n == "domshift" and not os.path.exists(config.ONEPOLE_CSV): print(f"   atlandı: {os.path.relpath(config.ONEPOLE_CSV, HERE)} yok (eski 1 kutuplu veri; bkz. README)"); continue
        if n == "latex": run_latex(); continue
        for s in scripts:
            if s == "generate_validation_4pole.py" and os.path.exists(config.VAL4_CSV): print("   atlandı: generate_validation_4pole.py (CSV mevcut)"); continue
            if run_script(s, env, config.LOG_DIR) != 0:
                fails.append(s)
                if not a.keep_going: print("\nDurduruldu. Düzeltip tekrar çalıştırın veya --keep-going kullanın."); return
    print(f"\nBitti: {(time.time()-T0)/60:.1f} dk; hatalı betik: {fails or 'yok'}")
if __name__ == "__main__": main()
