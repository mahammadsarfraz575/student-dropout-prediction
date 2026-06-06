"""
╔══════════════════════════════════════════════════════════════╗
║              config.py  —  Project Configuration             ║
║   No API keys needed — downloads directly from website       ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─── FOLDERS (auto-created) ───────────────────────────────────
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"
EXPORT_DIR  = "exports"
REPORT_DIR  = "reports"

# ─── MODEL SETTINGS ───────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
TARGET_COLUMN   = "final_result"

# ─── EARLY WARNING THRESHOLDS (days) ──────────────────────────
EARLY_DAY_1  = 30
EARLY_DAY_2  = 60
EARLY_DAY_3  = 100

# ─── PASS / FAIL MAPPING ──────────────────────────────────────
PASS_LABELS = {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
