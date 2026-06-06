"""
╔══════════════════════════════════════════════════════════════╗
║   ALTERNATIVE SETUP — Uses Kaggle Mirror (Backup)            ║
║   If Open University download fails, use this                ║
║   Command: python 00_setup_alternative.py                    ║
╚══════════════════════════════════════════════════════════════╝
"""
import subprocess, sys, os, zipfile

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
EXPORT_DIR = "exports"
REPORT_DIR = "reports"

REQUIRED_CSVS = [
    "courses.csv", "assessments.csv", "vle.csv",
    "studentInfo.csv", "studentRegistration.csv",
    "studentAssessment.csv", "studentVle.csv",
]

PACKAGES = [
    "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "scikit-learn", "xgboost", "streamlit",
    "openpyxl", "imbalanced-learn", "joblib", "kaggle",
]

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")
def ok(m):     print(f"  ✅  {m}")
def info(m):   print(f"  ℹ️   {m}")

# ─── STEP 1: INSTALL ───────────────────────────────────────────
def step1_install():
    banner("STEP 1 — Installing packages")
    for pkg in PACKAGES:
        print(f"  {pkg}...", end=" ", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                           capture_output=True)
        print("✅" if r.returncode == 0 else "⚠️")

# ─── STEP 2: FOLDERS ───────────────────────────────────────────
def step2_folders():
    banner("STEP 2 — Creating folders")
    for f in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, EXPORT_DIR, REPORT_DIR]:
        os.makedirs(f, exist_ok=True)
        ok(f"/{f}/")

# ─── STEP 3: DOWNLOAD FROM KAGGLE ─────────────────────────────
def step3_download():
    banner("STEP 3 — Downloading from Kaggle (Alternative)")
    
    already = [f for f in REQUIRED_CSVS if os.path.exists(os.path.join(DATA_DIR, f))]
    if len(already) == len(REQUIRED_CSVS):
        ok("All 7 CSV files present — skipping download")
        return

    info("Using Kaggle mirror dataset...")
    info("NOTE: This requires internet connection")
    
    try:
        # Import here after installation
        import kaggle
        
        # Download without authentication (public dataset)
        info("Downloading... (may take 2-3 minutes)")
        
        kaggle.api.dataset_download_files(
            "anlgrbz/student-demographics-open-university",
            path=DATA_DIR,
            unzip=True,
            quiet=False
        )
        
        ok("Download complete via Kaggle")
        
    except Exception as e:
        print(f"\n  ⚠️  Kaggle download also failed: {e}")
        print("""
  MANUAL DOWNLOAD REQUIRED:
  ─────────────────────────────────────────────────
  Option 1 (Recommended):
    1. Go to: https://analyse.kmi.open.ac.uk/open-dataset
    2. Click "Download OULAD"
    3. Extract ZIP to: data/
  
  Option 2 (Kaggle):
    1. Go to: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
    2. Click "Download"
    3. Extract ZIP to: data/
  
  Then re-run: python run_all.py
  ─────────────────────────────────────────────────
        """)

# ─── STEP 4: VERIFY ────────────────────────────────────────────
def step4_verify():
    banner("STEP 4 — Verifying files")
    import pandas as pd
    
    all_ok = True
    for fname in REQUIRED_CSVS:
        found = None
        for root, _, files in os.walk(DATA_DIR):
            if fname in files:
                found = os.path.join(root, fname)
                break
        
        if found:
            df = pd.read_csv(found)
            ok(f"{fname:35s} {len(df):>10,} rows × {df.shape[1]} cols")
            flat = os.path.join(DATA_DIR, fname)
            if found != flat:
                import shutil
                shutil.copy(found, flat)
        else:
            print(f"  ❌  {fname} NOT FOUND")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       🎓  OULAD Setup (Kaggle Alternative)                   ║
╚══════════════════════════════════════════════════════════════╝""")

    step1_install()
    step2_folders()
    step3_download()
    ok_flag = step4_verify()

    if ok_flag:
        banner("✅  SETUP COMPLETE")
        print("""
  Next: python run_all.py
        """)
    else:
        print("  ❌  Files missing — follow manual download above")
