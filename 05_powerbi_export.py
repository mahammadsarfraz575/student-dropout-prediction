"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 5 — Power BI Export                                   ║
║   Generates a multi-sheet Excel file ready for Power BI      ║
║   Command: python 05_powerbi_export.py                       ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd, numpy as np, os, warnings
from config import DATA_DIR, OUTPUT_DIR, EXPORT_DIR

warnings.filterwarnings("ignore")
os.makedirs(EXPORT_DIR, exist_ok=True)

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")
def ok(m):     print(f"  OK  {m}")

def load(f):
    p=os.path.join(DATA_DIR,f)
    return pd.read_csv(p) if os.path.exists(p) else None

def load_out(f):
    p=os.path.join(OUTPUT_DIR,f)
    return pd.read_csv(p) if os.path.exists(p) else None

banner("Loading data")
master       = load_out("master_dataset.csv")
student_info = load("studentInfo.csv")
student_vle  = load("studentVle.csv")
assessments  = load("assessments.csv")
student_assess=load("studentAssessment.csv")
student_reg  = load("studentRegistration.csv")
vle          = load("vle.csv")
courses      = load("courses.csv")

if master is None:
    print("  master_dataset.csv not found — run 02_feature_engineering.py first")
    exit(1)

banner("Building Power BI sheets")
out_path = os.path.join(EXPORT_DIR,"OULAD_PowerBI_Dashboard.xlsx")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    # ── Sheet 1: Raw master (all features) ───────────────────
    master.to_excel(writer, sheet_name="Master_Data", index=False)
    ok("Sheet: Master_Data")

    # ── Sheet 2: Result summary by demographics ───────────────
    summary = master.groupby(["gender","age_band","highest_education",
                               "final_result"]).size().reset_index(name="count")
    summary["pass_flag"] = summary["final_result"].isin(["Pass","Distinction"]).astype(int)
    summary.to_excel(writer, sheet_name="Result_Summary", index=False)
    ok("Sheet: Result_Summary")

    # ── Sheet 3: Module performance ───────────────────────────
    module_perf = master.groupby(["code_module","code_presentation"]).agg(
        total_students=("id_student","count"),
        pass_count=("target","sum"),
        avg_clicks=("total_clicks","mean"),
        avg_score=("avg_assessment_score","mean"),
        withdrawal_count=("withdrew","sum") if "withdrew" in master.columns
                          else ("target","count")
    ).reset_index()
    module_perf["pass_rate_pct"] = (module_perf["pass_count"]/
                                     module_perf["total_students"]*100).round(1)
    module_perf.to_excel(writer, sheet_name="Module_Performance", index=False)
    ok("Sheet: Module_Performance")

    # ── Sheet 4: VLE engagement bands ────────────────────────
    if "total_clicks" in master.columns:
        master["engagement_band"] = pd.cut(
            master["total_clicks"],
            bins=[0,200,500,1000,2000,999999],
            labels=["Very Low (0-200)","Low (200-500)","Medium (500-1000)",
                    "High (1000-2000)","Very High (2000+)"]
        )
        eng = master.groupby("engagement_band").agg(
            students=("id_student","count"),
            pass_rate=("target",lambda x: x.mean()*100)
        ).reset_index()
        eng.to_excel(writer, sheet_name="Engagement_Bands", index=False)
        ok("Sheet: Engagement_Bands")

    # ── Sheet 5: IMD deprivation analysis ─────────────────────
    if "imd_band" in master.columns:
        imd = master.groupby("imd_band").agg(
            students=("id_student","count"),
            pass_rate=("target",lambda x: x.mean()*100),
            avg_clicks=("total_clicks","mean"),
            avg_score=("avg_assessment_score","mean")
        ).reset_index().dropna(subset=["imd_band"])
        imd.to_excel(writer, sheet_name="IMD_Deprivation", index=False)
        ok("Sheet: IMD_Deprivation")

    # ── Sheet 6: Daily VLE time series ────────────────────────
    if student_vle is not None:
        daily = student_vle.groupby("date").agg(
            total_clicks=("sum_click","sum"),
            unique_students=("id_student","nunique")
        ).reset_index().sort_values("date")
        daily.to_excel(writer, sheet_name="Daily_VLE_Activity", index=False)
        ok("Sheet: Daily_VLE_Activity")

    # ── Sheet 7: Assessment type performance ──────────────────
    if student_assess is not None and assessments is not None:
        am = student_assess.merge(
            assessments[["id_assessment","assessment_type","weight"]],
            on="id_assessment",how="left"
        ).merge(student_info[["id_student","final_result"]],
                on="id_student",how="left")
        at_perf = am.groupby(["assessment_type","final_result"])["score"].agg(
            count="count", mean="mean", median="median", std="std"
        ).round(2).reset_index()
        at_perf.to_excel(writer, sheet_name="Assessment_Performance", index=False)
        ok("Sheet: Assessment_Performance")

    # ── Sheet 8: Early warning (day-30 snapshot) ──────────────
    if "clicks_day30" in master.columns:
        ew = master[["id_student","gender","age_band","highest_education",
                     "imd_band","code_module","clicks_day30","active_days",
                     "avg_assessment_score","final_result","target"]].copy()
        ew["risk_flag"] = ((ew["clicks_day30"]<200) &
                            (ew["avg_assessment_score"]<50)).astype(int)
        ew.to_excel(writer, sheet_name="Early_Warning_Day30", index=False)
        ok("Sheet: Early_Warning_Day30")

    # ── Sheet 9: Withdrawal timeline ─────────────────────────
    if student_reg is not None:
        w = student_reg[student_reg["date_unregistration"].notna()].copy()
        w["date_unregistration"] = pd.to_numeric(w["date_unregistration"],errors="coerce")
        w["withdrawal_window"] = pd.cut(
            w["date_unregistration"],
            bins=[0,30,60,100,200,999],
            labels=["0-30 days","31-60 days","61-100 days","101-200 days","200+ days"]
        )
        wt = w.groupby("withdrawal_window").size().reset_index(name="withdrawals")
        wt.to_excel(writer, sheet_name="Withdrawal_Timeline", index=False)
        ok("Sheet: Withdrawal_Timeline")

    # ── Sheet 10: KPI summary card ────────────────────────────
    kpi = pd.DataFrame([{
        "Total Students":       len(master),
        "Pass Rate (%)":        round(master["target"].mean()*100,1),
        "Avg VLE Clicks":       round(master["total_clicks"].mean(),0),
        "Avg Assessment Score": round(master["avg_assessment_score"].mean(),1),
        "Withdrawal Rate (%)":  round(master["withdrew"].mean()*100,1) if "withdrew" in master.columns else "N/A",
        "Distinction Count":    (master["final_result"]=="Distinction").sum(),
        "At Risk (Fail+Withdraw)": (master["final_result"].isin(["Fail","Withdrawn"])).sum(),
    }])
    kpi.to_excel(writer, sheet_name="KPI_Summary", index=False)
    ok("Sheet: KPI_Summary")

ok(f"\nSaved: {out_path}")
print(f"""
  Power BI Instructions:
  ──────────────────────────────────────────────────────
  1. Open Power BI Desktop (free from microsoft.com)
  2. Click "Get Data" → "Excel Workbook"
  3. Open: {out_path}
  4. Select all 10 sheets → Load
  5. Create relationships:
       Master_Data[id_student] ↔ Early_Warning_Day30[id_student]
       Master_Data[code_module] ↔ Module_Performance[code_module]
  6. Build visuals from the field list on the right
  7. Publish → Power BI Service (free tier) to share online
  ──────────────────────────────────────────────────────
""")
banner("POWER BI EXPORT COMPLETE")
