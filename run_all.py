"""
╔══════════════════════════════════════════════════════════════╗
║   RUN ALL — Master runner: executes every step in order      ║
║   Command: python run_all.py                                 ║
║                                                              ║
║   Steps:                                                     ║
║   0. Setup (install + download via Kaggle API)               ║
║   1. EDA                                                     ║
║   2. Feature Engineering                                     ║
║   3. ML Models                                               ║
║   5. Power BI Export                                         ║
║   Then: streamlit run 04_dashboard.py                        ║
╚══════════════════════════════════════════════════════════════╝
"""
import subprocess, sys, time, os

STEPS = [
    ("00_setup.py",               "Setup & Download Data via API"),
    ("01_eda.py",                  "Exploratory Data Analysis"),
    ("02_feature_engineering.py", "Feature Engineering"),
    ("03_ml_models.py",           "ML Model Training"),
    ("05_powerbi_export.py",      "Power BI Excel Export"),
]

def run_step(script, label):
    print(f"\n{'#'*62}")
    print(f"#  RUNNING: {label}")
    print(f"#  Script : {script}")
    print(f"{'#'*62}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  DONE in {elapsed:.0f}s")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode})")
        return False

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       OULAD Full Data Science Pipeline — Running All         ║
╚══════════════════════════════════════════════════════════════╝
Before starting: make sure you filled in config.py !
  KAGGLE_USERNAME = "your_kaggle_username"
  KAGGLE_API_KEY  = "your_api_key"
""")

    failed = []
    for script, label in STEPS:
        if not os.path.exists(script):
            print(f"  SKIP (not found): {script}")
            continue
        ok = run_step(script, label)
        if not ok:
            failed.append(script)
            ans = input(f"\n  {script} failed. Continue anyway? (y/n): ")
            if ans.strip().lower() != "y":
                print("  Stopping pipeline.")
                break

    print(f"""
{'='*62}
  PIPELINE COMPLETE
{'='*62}
  Results saved in /outputs/
  Models saved in  /models/
  Power BI file:   /exports/OULAD_PowerBI_Dashboard.xlsx

  FINAL STEP — Launch the Streamlit dashboard:
  streamlit run 04_dashboard.py

  Then open your browser at: http://localhost:8501
{'='*62}
""")
    if failed:
        print(f"  Steps that had errors: {failed}")
        print("  Review the output above and fix before retrying.")
