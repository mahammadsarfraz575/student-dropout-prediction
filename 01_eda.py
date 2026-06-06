"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 1 — EDA: Explore all 7 OULAD tables                   ║
║   Command: python 01_eda.py                                  ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px, os, warnings
from config import DATA_DIR, OUTPUT_DIR

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid")

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")
def ok(m):     print(f"  OK  {m}")

def load(f): return pd.read_csv(os.path.join(DATA_DIR, f))

banner("Loading 7 OULAD tables")
courses        = load("courses.csv")
assessments    = load("assessments.csv")
vle            = load("vle.csv")
student_info   = load("studentInfo.csv")
student_reg    = load("studentRegistration.csv")
student_assess = load("studentAssessment.csv")
student_vle    = load("studentVle.csv")

for name,df in [("courses",courses),("assessments",assessments),("vle",vle),
                ("studentInfo",student_info),("studentRegistration",student_reg),
                ("studentAssessment",student_assess),("studentVle",student_vle)]:
    print(f"  {name:25s} -> {len(df):>10,} rows x {df.shape[1]} cols")

# A. Student Info
banner("A — Student Info")
result_dist = student_info["final_result"].value_counts()
print("Final Result:\n", result_dist.to_string())

edu_result = pd.crosstab(student_info["highest_education"],
                         student_info["final_result"], normalize="index")*100
imd_result = pd.crosstab(student_info["imd_band"],
                         student_info["final_result"], normalize="index")*100
disability_pass = student_info.groupby("disability")["final_result"].apply(
    lambda x: (x.isin(["Pass","Distinction"])).sum()/len(x)*100).round(1)
print("Pass rate by disability:\n", disability_pass.to_string())

# B. VLE
banner("B — VLE Clickstream")
clicks_per_student = (student_vle.groupby("id_student")["sum_click"]
                      .sum().reset_index()
                      .rename(columns={"sum_click":"total_clicks"}))
clicks_result = clicks_per_student.merge(
    student_info[["id_student","final_result"]], on="id_student", how="left")
print("Avg clicks by result:\n",
      clicks_result.groupby("final_result")["total_clicks"].mean()
      .sort_values(ascending=False).round(0).to_string())

# C. Assessments
banner("C — Assessments")
assess_m = (student_assess
    .merge(assessments[["id_assessment","assessment_type","weight"]],
           on="id_assessment", how="left")
    .merge(student_info[["id_student","final_result"]],
           on="id_student", how="left"))
print("Score by result:\n",
      assess_m.groupby("final_result")["score"].mean().round(1).to_string())

# D. Registration
banner("D — Registration")
withdrew  = student_reg[student_reg["date_unregistration"].notna()]
w_days    = withdrew["date_unregistration"].astype(float)
early     = (w_days<=30).sum()
print(f"Withdrawals: {len(withdrew):,}  Early: {early:,}")

# E. Plots
banner("E — Saving plots")
colors={"Pass":"#2ecc71","Fail":"#e74c3c","Distinction":"#3498db","Withdrawn":"#95a5a6"}
fig,axes=plt.subplots(3,3,figsize=(20,14))
fig.suptitle("OULAD Full EDA",fontsize=16,fontweight="bold",y=1.01)

# 1
ax=axes[0,0]
result_dist.plot(kind="bar",ax=ax,color=[colors.get(r,"grey") for r in result_dist.index])
ax.set_title("Final Result Distribution"); ax.tick_params(axis="x",rotation=30)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",(p.get_x()+p.get_width()/2,p.get_height()),
                ha="center",va="bottom",fontsize=8)

# 2
ax=axes[0,1]
pd.crosstab(student_info["gender"],student_info["final_result"]).plot(
    kind="bar",stacked=True,ax=ax,colormap="Set2")
ax.set_title("Gender vs Result"); ax.tick_params(axis="x",rotation=0); ax.legend(fontsize=7)

# 3
ax=axes[0,2]
pd.crosstab(student_info["age_band"],student_info["final_result"]).plot(
    kind="bar",stacked=True,ax=ax,colormap="Paired")
ax.set_title("Age Band vs Result"); ax.tick_params(axis="x",rotation=30); ax.legend(fontsize=7)

# 4
ax=axes[1,0]
sns.heatmap(edu_result.round(1),annot=True,fmt=".0f",cmap="YlOrRd",ax=ax,
            cbar=False,linewidths=0.5)
ax.set_title("Education vs Result (%)"); ax.tick_params(axis="x",rotation=30)

# 5
ax=axes[1,1]
imd_order=[c for c in ["0-10%","10-20%","20-30%","30-40%","40-50%",
                        "50-60%","60-70%","70-80%","80-90%","90-100%"]
           if c in imd_result.index]
imd_s=imd_result.reindex(imd_order).dropna()
if len(imd_s):
    pass_pct=(imd_s.get("Pass",0)+imd_s.get("Distinction",0))
    pass_pct.plot(kind="bar",ax=ax,color="#2ecc71",alpha=0.85)
ax.set_title("IMD Band vs Pass Rate (%)"); ax.tick_params(axis="x",rotation=40)

# 6
ax=axes[1,2]
order=[o for o in ["Distinction","Pass","Fail","Withdrawn"]
       if o in clicks_result["final_result"].unique()]
clean=clicks_result[clicks_result["total_clicks"]<
                    clicks_result["total_clicks"].quantile(0.99)]
sns.boxplot(data=clean,x="final_result",y="total_clicks",
            order=order,palette="Set2",ax=ax)
ax.set_title("VLE Clicks vs Result"); ax.tick_params(axis="x",rotation=20)

# 7
ax=axes[2,0]
for res,grp in assess_m.groupby("final_result"):
    grp["score"].dropna().hist(bins=30,ax=ax,alpha=0.5,
                               label=res,color=colors.get(res,"grey"))
ax.set_title("Score Distribution by Result"); ax.legend(fontsize=7)

# 8
ax=axes[2,1]
w_days.dropna().astype(float).hist(bins=50,ax=ax,color="#e74c3c",alpha=0.7)
ax.axvline(30,color="orange",linestyle="--",linewidth=2,label="Day 30")
ax.axvline(100,color="blue",linestyle="--",linewidth=2,label="Day 100")
ax.set_title("Withdrawal Day Distribution"); ax.legend(fontsize=8)

# 9
ax=axes[2,2]
disability_pass.plot(kind="bar",ax=ax,color=["#3498db","#e74c3c"],alpha=0.85)
ax.set_title("Pass Rate by Disability (%)"); ax.set_ylim(0,100)

plt.tight_layout()
out=os.path.join(OUTPUT_DIR,"01_eda_full.png")
plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
ok(f"Saved: {out}")

# Sunburst
fig2=px.sunburst(student_info,path=["gender","age_band","final_result"],
    title="Demographics -> Final Result")
fig2.write_html(os.path.join(OUTPUT_DIR,"01_eda_sunburst.html"))
ok("Saved: 01_eda_sunburst.html")

# VLE time series
daily=student_vle.groupby("date")["sum_click"].sum().reset_index()
daily.columns=["day","clicks"]
px.line(daily.sort_values("day"),x="day",y="clicks",
        title="Daily VLE Clicks").write_html(os.path.join(OUTPUT_DIR,"01_vle_timeseries.html"))
ok("Saved: 01_vle_timeseries.html")

# Save processed CSVs
student_info.to_csv(os.path.join(OUTPUT_DIR,"eda_student_info.csv"),index=False)
clicks_result.to_csv(os.path.join(OUTPUT_DIR,"eda_clicks_result.csv"),index=False)
assess_m.to_csv(os.path.join(OUTPUT_DIR,"eda_assess_merged.csv"),index=False)

banner("EDA COMPLETE -> next: python 02_feature_engineering.py")
