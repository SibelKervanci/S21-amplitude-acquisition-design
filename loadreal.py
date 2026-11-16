import numpy as np, pandas as pd, glob, os, warnings
warnings.filterwarnings('ignore')
E0 = 8.854e-12
def read_one(path):
    df = pd.read_excel(path, engine="odf", header=None)
    f = df.iloc[2:, 0].astype(float).values
    codes = [str(x) for x in df.iloc[1, 1:].dropna().values]
    M = df.iloc[2:, 1:1+len(codes)].astype(float).values
    keep = [i for i, c in enumerate(codes) if not any(t in c.lower() for t in ("avg", "stdev"))]   # ortalama/std sütunlarını at
    return f, [codes[i] for i in keep], M[:, keep]
def _find_pairs(root):
    """root altında (alt klasörler dahil) <taban>_e<'>.ods ve <taban>_e<''>.ods çiftlerini bulur.
    Kesme işareti yerine farklı karakter (’ ′ _ vb.) olsa da çalışır: ekin uzunluğuna göre e' (1) ve e'' (2) ayrılır."""
    import re
    pat = re.compile(r"^(?P<base>.+)_e(?P<sfx>[^A-Za-z0-9]{1,3})\.ods$")
    found, all_ods = {}, []
    for dp, _, fn in os.walk(root):
        for f in sorted(fn):
            if f.lower().endswith(".ods"): all_ods.append(f)
            m = pat.match(f)
            if m: found.setdefault(m.group("base"), {}).setdefault(len(m.group("sfx")), os.path.join(dp, f))
    pairs = {b: (d[1], d[2]) for b, d in found.items() if 1 in d and 2 in d}
    return pairs, all_ods
def _extract_from_zip(root):
    """root ve bir üst klasörde *.zip arar; içinde .ods varsa root'a çıkarır (CORA.RDR zip'i: doi-10_34810-data2340.zip)."""
    import zipfile, glob
    cands = glob.glob(os.path.join(root, "*.zip")) + glob.glob(os.path.join(os.path.dirname(os.path.abspath(root)), "*.zip"))
    for zp in cands:
        try:
            with zipfile.ZipFile(zp) as z:
                members = [m for m in z.namelist() if m.lower().endswith(".ods")]
                if not members: continue
                os.makedirs(root, exist_ok=True)
                for m in members:
                    with z.open(m) as src, open(os.path.join(root, os.path.basename(m)), "wb") as dst: dst.write(src.read())
            print(f"[loadreal] {len(members)} .ods files extracted from {os.path.basename(zp)} into {root}")
            return len(members)
        except zipfile.BadZipFile: continue
    return 0
def load_all(root="."):
    if not os.path.isdir(root): raise FileNotFoundError(f"Gerçek veri klasörü yok: {root}\n  -> CORA.RDR .ods dosyalarını bu klasöre koyun (README, Bölüm 1).")
    pairs, all_ods = _find_pairs(root)
    if not pairs and _extract_from_zip(root): pairs, all_ods = _find_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"{root} içinde (alt klasörler dahil) beklenen dosya çiftleri bulunamadı.\n  Bulunan .ods dosya sayısı: {len(all_ods)}; ilk birkaçı: {all_ods[:4]}\n"
                                "  Çözüm: CORA.RDR zip'ini (doi-10_34810-data2340.zip) proje klasörüne veya real_data\\ içine koyun (otomatik açılır) ya da içindeki .ods dosyalarını real_data\\ içine çıkarın.\n"
                                "  Beklenen ad biçimi: <Doku>_<histoloji>_e'.ods ve <Doku>_<histoloji>_e''.ods (ör. Heart_standard_e'.ods).")
    out = {}
    for base in sorted(pairs):
        p1, p2 = pairs[base]
        try: f, c1, er = read_one(p1); f2, c2, ei = read_one(p2)
        except ImportError as e: raise ImportError("ODS okumak için odfpy gerekir:  pip install odfpy") from e
        assert np.allclose(f, f2) and er.shape == ei.shape, (base, er.shape, ei.shape)   # örnekler konuma göre eşlenir
        out[base] = {"f": f, "codes": c1, "eps_re": er, "eps_im": ei}
    return out
def at_freqs(v, f_ghz):
    """Her örnek için ε′ ve σ_eff'i istenen frekanslarda (log-f'de doğrusal interpolasyon) döndürür: (n, len(f_ghz))."""
    lf = np.log(v["f"]); lt = np.log(np.asarray(f_ghz)*1e9)
    er = np.array([np.interp(lt, lf, v["eps_re"][:, j]) for j in range(v["eps_re"].shape[1])])
    ei = np.array([np.interp(lt, lf, v["eps_im"][:, j]) for j in range(v["eps_im"].shape[1])])
    return er, ei*2*np.pi*np.asarray(f_ghz)*1e9*E0
MAP = {"Heart_standard":"Heart", "KidneyCortex_standard":"Kidney", "Spleen_standard":"Spleen", "Liver_standard":"Liver",
       "SkeletalMuscle_standard":"Muscle", "BrainGreyMatter_standard":"Brain_grey", "BrainWhiteMatter_standard":"Brain_white"}
