# config.py — tüm betikler için ortak yol ve mod ayarları
import os
ROOT = os.path.abspath(os.environ.get("REPRO_DIR", os.path.dirname(os.path.abspath(__file__))))
SMOKE = os.environ.get("SMOKE", "0") == "1"          # SMOKE=1: çok küçük boyutlarla hızlı deneme (sonuçlar makaleye girmez)
os.environ["GABRIEL_DIR"] = ROOT                      # generate_dataset_4pole.py bu klasörün altına "gabriel_datasets" yazar
DATA_DIR  = os.path.join(ROOT, "gabriel_datasets")
REAL_DIR  = os.environ.get("REAL_DATA_DIR", os.path.join(ROOT, "real_data"))   # CORA.RDR .ods dosyaları buraya
OUT_ROOT  = os.path.join(ROOT, "outputs_smoke" if SMOKE else "outputs")
STATS_DIR = os.path.join(OUT_ROOT, "stats_out")       # preds.npz, stats.json, preds_mlp.npz
WORK_DIR  = os.path.join(OUT_ROOT, "work")            # ara dosyalar (npz, csv)
LOG_DIR   = os.path.join(OUT_ROOT, "logs")
PAPER_DIR = os.path.join(ROOT, "paper")
FIG_DIR   = os.path.join(OUT_ROOT, "figs") if SMOKE else os.path.join(PAPER_DIR, "figs")      # smoke makale klasörünü bozmaz
AUTO_DIR  = os.path.join(OUT_ROOT, "auto") if SMOKE else os.path.join(PAPER_DIR, "auto")
TRAIN_CSV   = os.path.join(DATA_DIR, "Dataset_4pole_train.csv")
ONEPOLE_CSV = os.path.join(DATA_DIR, "Dataset_ITIS_ranges_v2.csv")          # yalnız domshift.py (eski 1 kutuplu veri)
VAL4_CSV    = os.path.join(DATA_DIR, "Dataset_ITIS_4pole_validation.csv")   # yalnız domshift.py
for _d in (DATA_DIR, STATS_DIR, WORK_DIR, LOG_DIR, FIG_DIR, AUTO_DIR): os.makedirs(_d, exist_ok=True)
def n(full, smoke):
    """Tam modda 'full', smoke modunda 'smoke' döndürür."""
    return smoke if SMOKE else full
