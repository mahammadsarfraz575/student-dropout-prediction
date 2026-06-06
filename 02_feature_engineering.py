"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 2 — Feature Engineering                               ║
║   Merges all 7 tables → master ML-ready dataset              ║
║   Command: python 02_feature_engineering.py                  ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd, numpy as np, os, warnings
from config import DATA_DIR, OUTPUT_DIR, PASS_LABELS, EARLY_DAY_1, EARLY_DAY_2, EARLY_DAY_3

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")
def ok(m):     print(f"  OK  {m}")

def load(f): return pd.read_csv(os.path.join(DATA_DIR, f))

banner("Loading tables")
student_info   = load("studentInfo.csv")
student_reg    = load("studentRegistration.csv")
student_assess = load("studentAssessment.csv")
assessments    = load("assessments.csv")
student_vle    = load("studentVle.csv")
vle            = load("vle.csv")
courses        = load("courses.csv")
print(f"  studentInfo rows: {len(student_info):,}")

# ─── A. VLE FEATURES ───────────────────────────────────────────
banner("A — VLE engagement features")

# Total clicks per student
total_clicks = (student_vle.groupby("id_student")["sum_click"]
                .sum().reset_index().rename(columns={"sum_click":"total_clicks"}))

# Unique days active
active_days = (student_vle.groupby("id_student")["date"]
               .nunique().reset_index().rename(columns={"date":"active_days"}))

# Clicks in early windows
def early_clicks(days_limit):
    sub = student_vle[student_vle["date"] <= days_limit]
    return (sub.groupby("id_student")["sum_click"]
            .sum().reset_index()
            .rename(columns={"sum_click":f"clicks_day{days_limit}"}))

clicks_30  = early_clicks(EARLY_DAY_1)
clicks_60  = early_clicks(EARLY_DAY_2)
clicks_100 = early_clicks(EARLY_DAY_3)

# Session consistency: std of daily clicks
session_std = (student_vle.groupby("id_student")["sum_click"]
               .std().fillna(0).reset_index()
               .rename(columns={"sum_click":"click_std"}))

# Unique material types accessed
vle_types = student_vle.merge(vle[["id_site","activity_type"]], on="id_site", how="left")
unique_types = (vle_types.groupby("id_student")["activity_type"]
                .nunique().reset_index()
                .rename(columns={"activity_type":"unique_activity_types"}))

# Last active day (recency)
last_day = (student_vle.groupby("id_student")["date"]
            .max().reset_index().rename(columns={"date":"last_active_day"}))

ok("VLE features built")

# ─── B. ASSESSMENT FEATURES ────────────────────────────────────
banner("B — Assessment features")

assess_m = student_assess.merge(
    assessments[["id_assessment","assessment_type","weight","date"]],
    on="id_assessment", how="left")

# Average score per student
avg_score = (assess_m.groupby("id_student")["score"]
             .mean().reset_index().rename(columns={"score":"avg_assessment_score"}))

# Score std (consistency)
score_std = (assess_m.groupby("id_student")["score"]
             .std().fillna(0).reset_index()
             .rename(columns={"score":"score_std"}))

# Number of assessments submitted
num_submit = (assess_m.groupby("id_student")["id_assessment"]
              .count().reset_index().rename(columns={"id_assessment":"num_assessments_submitted"}))

# TMA average (continuous assessment)
tma = assess_m[assess_m["assessment_type"]=="TMA"]
tma_avg = (tma.groupby("id_student")["score"]
           .mean().reset_index().rename(columns={"score":"tma_avg_score"}))

# Exam score
exam = assess_m[assess_m["assessment_type"]=="Exam"]
exam_avg = (exam.groupby("id_student")["score"]
            .mean().reset_index().rename(columns={"score":"exam_avg_score"}))

# Late submissions
if "date_submitted" in assess_m.columns and "date" in assess_m.columns:
    assess_m["is_late"] = assess_m["date_submitted"] > assess_m["date"]
    late_count = (assess_m.groupby("id_student")["is_late"]
                  .sum().reset_index().rename(columns={"is_late":"late_submissions"}))
else:
    late_count = pd.DataFrame({"id_student":assess_m["id_student"].unique(),"late_submissions":0})

ok("Assessment features built")

# ─── C. REGISTRATION FEATURES ─────────────────────────────────
banner("C — Registration features")

student_reg["withdrew"]        = student_reg["date_unregistration"].notna().astype(int)
student_reg["withdrawal_day"]  = pd.to_numeric(student_reg["date_unregistration"],errors="coerce").fillna(999)
student_reg["early_withdraw"]  = (student_reg["withdrawal_day"] <= EARLY_DAY_1).astype(int)
student_reg["registered_late"] = (pd.to_numeric(student_reg["date_registration"],errors="coerce") > 0).astype(int)

reg_features = (student_reg.groupby("id_student")
                .agg(num_modules_registered=("code_presentation","count"),
                     withdrew=("withdrew","max"),
                     withdrawal_day=("withdrawal_day","min"),
                     early_withdraw=("early_withdraw","max"),
                     registered_late=("registered_late","max"))
                .reset_index())
ok("Registration features built")

# ─── D. MERGE ALL → MASTER DATASET ────────────────────────────
banner("D — Merging everything")

master = student_info.copy()

for df, key in [
    (total_clicks,    "id_student"),
    (active_days,     "id_student"),
    (clicks_30,       "id_student"),
    (clicks_60,       "id_student"),
    (clicks_100,      "id_student"),
    (session_std,     "id_student"),
    (unique_types,    "id_student"),
    (last_day,        "id_student"),
    (avg_score,       "id_student"),
    (score_std,       "id_student"),
    (num_submit,      "id_student"),
    (tma_avg,         "id_student"),
    (exam_avg,        "id_student"),
    (late_count,      "id_student"),
    (reg_features,    "id_student"),
]:
    master = master.merge(df, on=key, how="left")

print(f"  Master shape before fill: {master.shape}")

# Fill numeric NaN with 0
num_cols = master.select_dtypes(include="number").columns
master[num_cols] = master[num_cols].fillna(0)

ok(f"Master shape: {master.shape}")

# ─── E. ENCODE TARGET & CATEGORICAL ───────────────────────────
banner("E — Encoding")

master["target"] = master["final_result"].map(PASS_LABELS)

cat_cols = ["gender","region","highest_education","imd_band",
            "age_band","disability","code_module","code_presentation"]

for col in cat_cols:
    if col in master.columns:
        master[col+"_enc"] = pd.Categorical(master[col]).codes

# Drop raw string cols for ML (keep for display)
feature_cols = (
    [c+"_enc" for c in cat_cols if c in master.columns]
    + ["total_clicks","active_days",
       f"clicks_day{EARLY_DAY_1}",f"clicks_day{EARLY_DAY_2}",f"clicks_day{EARLY_DAY_3}",
       "click_std","unique_activity_types","last_active_day",
       "avg_assessment_score","score_std","num_assessments_submitted",
       "tma_avg_score","exam_avg_score","late_submissions",
       "num_modules_registered","withdrawal_day","early_withdraw",
       "studied_credits","num_of_prev_attempts"]
)
feature_cols = [c for c in feature_cols if c in master.columns]

print(f"  Features: {len(feature_cols)}")
print(f"  Target distribution:\n  {master['target'].value_counts().to_string()}")

# ─── F. SAVE ──────────────────────────────────────────────────
master.to_csv(os.path.join(OUTPUT_DIR,"master_dataset.csv"), index=False)
pd.Series(feature_cols).to_csv(os.path.join(OUTPUT_DIR,"feature_cols.csv"), index=False, header=False)
ok("Saved: master_dataset.csv")
ok("Saved: feature_cols.csv")

# ─── G. EARLY WARNING SNAPSHOT ────────────────────────────────
banner("G — Early Warning Dataset (day-30 only)")
ew_cols = [f"clicks_day{EARLY_DAY_1}","active_days",
           "avg_assessment_score","early_withdraw","target"] + \
          [c+"_enc" for c in cat_cols if c in master.columns]
ew_cols = [c for c in ew_cols if c in master.columns]
master[ew_cols].to_csv(os.path.join(OUTPUT_DIR,"early_warning_dataset.csv"), index=False)
ok("Saved: early_warning_dataset.csv")

banner("FEATURE ENGINEERING COMPLETE -> next: python 03_ml_models.py")
