"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 4 — Streamlit Dashboard (full interactive app)        ║
║   Command: streamlit run 04_dashboard.py                     ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import joblib, os, warnings
from config import OUTPUT_DIR, MODEL_DIR, PASS_LABELS

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OULAD Student Analytics",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0e1117}
.kpi{background:linear-gradient(135deg,#1a1f35,#232840);border-radius:12px;
     padding:18px 12px;text-align:center;border-left:4px solid #4f8ef7;margin:4px}
.kpi .label{color:#8892b0;font-size:12px;margin:0}
.kpi .value{color:#fff;font-size:28px;font-weight:700;margin:4px 0}
.kpi .delta{color:#00d9a3;font-size:11px}
.sec{font-size:20px;font-weight:700;color:#4f8ef7;
     border-bottom:2px solid #4f8ef7;padding-bottom:4px;margin:16px 0 12px}
.ibox{background:#1a1f35;border-radius:10px;padding:16px;
      border:1px solid #2d3555;color:#cdd6f4;font-size:14px;line-height:1.8}
</style>""", unsafe_allow_html=True)


# ─── DATA LOADERS ──────────────────────────────────────────────
@st.cache_data
def load_master():
    p = os.path.join(OUTPUT_DIR,"master_dataset.csv")
    if os.path.exists(p): return pd.read_csv(p)
    return None

@st.cache_data
def load_student_vle():
    from config import DATA_DIR
    p = os.path.join(DATA_DIR,"studentVle.csv")
    if os.path.exists(p): return pd.read_csv(p)
    return None

@st.cache_resource
def load_model():
    try:
        bpath = os.path.join(MODEL_DIR,"best_model.txt")
        name  = open(bpath).read().strip() if os.path.exists(bpath) else "random_forest"
        mpath = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(mpath):
            # try any pkl
            pkls = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
            if pkls: mpath = os.path.join(MODEL_DIR, pkls[0])
            else: return None, None
        model  = joblib.load(mpath)
        fcols  = pd.read_csv(os.path.join(OUTPUT_DIR,"feature_cols.csv"),
                              header=None)[0].tolist()
        return model, fcols
    except Exception as e:
        return None, None

master   = load_master()
vle_data = load_student_vle()
model, feature_cols = load_model()

if master is None:
    st.error("Run steps 00→03 first, then come back here.")
    st.stop()

# ─── SIDEBAR FILTERS ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/graduation-cap.png", width=60)
    st.markdown("## 🎛️ Filters")

    genders = st.multiselect("Gender", master["gender"].unique(),
                              default=list(master["gender"].unique()))
    ages    = st.multiselect("Age Band", master["age_band"].unique(),
                              default=list(master["age_band"].unique()))
    edus    = st.multiselect("Education Level",
                              master["highest_education"].unique(),
                              default=list(master["highest_education"].unique()))
    results = st.multiselect("Final Result",
                              master["final_result"].unique(),
                              default=list(master["final_result"].unique()))
    st.markdown("---")
    score_range = st.slider("Avg Assessment Score", 0, 100, (0, 100))

    st.markdown("---")
    st.markdown("### 📥 Downloads")
    st.download_button("⬇️ Download Filtered CSV",
                        master.to_csv(index=False), "filtered.csv", "text/csv")

fdf = master[
    master["gender"].isin(genders) &
    master["age_band"].isin(ages) &
    master["highest_education"].isin(edus) &
    master["final_result"].isin(results) &
    master["avg_assessment_score"].between(score_range[0], score_range[1])
]

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("# 🎓 Open University Learning Analytics")
st.caption(f"Showing **{len(fdf):,}** of **{len(master):,}** students")

# ─── KPIs ──────────────────────────────────────────────────────
colors_kpi=["#4f8ef7","#00d9a3","#ff6b9d","#ffb347","#a78bfa","#38bdf8"]
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    ("Total Students",   f"{len(fdf):,}",            k1, colors_kpi[0]),
    ("Pass Rate",        f"{(fdf['target']==1).mean()*100:.1f}%", k2, colors_kpi[1]),
    ("Avg VLE Clicks",   f"{fdf['total_clicks'].mean():,.0f}", k3, colors_kpi[2]),
    ("Avg Assess Score", f"{fdf['avg_assessment_score'].mean():.1f}", k4, colors_kpi[3]),
    ("Avg Active Days",  f"{fdf['active_days'].mean():.0f}", k5, colors_kpi[4]),
    ("Withdrawal Rate",  f"{fdf['withdrew'].mean()*100:.1f}%" if 'withdrew' in fdf else "N/A", k6, colors_kpi[5]),
]
for label,val,col,clr in kpis:
    col.markdown(f"""<div class="kpi" style="border-left-color:{clr}">
    <p class="label">{label}</p><p class="value">{val}</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 Overview","🔍 Demographics","📈 Engagement",
    "📝 Assessments","🤖 Predict"
])

TPL = "plotly_dark"
def fig_bg(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ════════ TAB 1 — OVERVIEW ════════════════════════════════════
with t1:
    st.markdown('<p class="sec">📊 Result Distribution & Module Analysis</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        clr_map={"Pass":"#2ecc71","Distinction":"#3498db","Fail":"#e74c3c","Withdrawn":"#95a5a6"}
        res_cnt = fdf["final_result"].value_counts().reset_index()
        res_cnt.columns=["result","count"]
        fig=px.bar(res_cnt,x="result",y="count",color="result",
                   color_discrete_map=clr_map,template=TPL,
                   title="Final Result Distribution",text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig_bg(fig), use_container_width=True)

    with c2:
        module_pass = fdf.groupby("code_module").apply(
            lambda x: (x["final_result"].isin(["Pass","Distinction"])).mean()*100
        ).reset_index(); module_pass.columns=["module","pass_rate"]
        fig2=px.bar(module_pass.sort_values("pass_rate",ascending=True),
                    x="pass_rate",y="module",orientation="h",
                    title="Pass Rate by Module (%)",template=TPL,
                    color="pass_rate",color_continuous_scale="Tealgrn")
        st.plotly_chart(fig_bg(fig2), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        fig3=px.pie(fdf["final_result"].value_counts().reset_index(),
                    names="final_result",values="count",
                    color="final_result",color_discrete_map=clr_map,
                    title="Result Share",template=TPL)
        st.plotly_chart(fig_bg(fig3), use_container_width=True)
    with c4:
        pres_pass=fdf.groupby("code_presentation").apply(
            lambda x: (x["final_result"].isin(["Pass","Distinction"])).mean()*100
        ).reset_index(); pres_pass.columns=["presentation","pass_rate"]
        fig4=px.line(pres_pass,x="presentation",y="pass_rate",markers=True,
                     title="Pass Rate by Presentation",template=TPL)
        st.plotly_chart(fig_bg(fig4), use_container_width=True)

# ════════ TAB 2 — DEMOGRAPHICS ════════════════════════════════
with t2:
    st.markdown('<p class="sec">🔍 Demographic Breakdown</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)

    with c1:
        df_g=pd.crosstab(fdf["gender"],fdf["final_result"]).reset_index().melt("gender")
        fig=px.bar(df_g,x="gender",y="value",color="final_result",barmode="group",
                   title="Gender vs Result",template=TPL,
                   color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                        "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig), use_container_width=True)

    with c2:
        df_a=pd.crosstab(fdf["age_band"],fdf["final_result"]).reset_index().melt("age_band")
        fig2=px.bar(df_a,x="age_band",y="value",color="final_result",
                    barmode="stack",title="Age Band vs Result",template=TPL,
                    color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                         "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig2), use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        edu_order=["No Formal quals","Lower Than A Level","A Level or Equivalent",
                   "HE Qualification","Post Graduate Qualification"]
        edu_ok=[e for e in edu_order if e in fdf["highest_education"].unique()]
        edu_other=[e for e in fdf["highest_education"].unique() if e not in edu_ok]
        edu_order_final = edu_ok + edu_other
        df_e = fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        df_e.columns=["education","pass_rate"]
        fig3=px.bar(df_e.sort_values("pass_rate",ascending=True),
                    x="pass_rate",y="education",orientation="h",
                    title="Pass Rate by Education (%)",template=TPL,
                    color="pass_rate",color_continuous_scale="Blues")
        st.plotly_chart(fig_bg(fig3), use_container_width=True)

    with c4:
        if "imd_band" in fdf.columns:
            imd_pass=fdf.groupby("imd_band")["target"].mean().mul(100).reset_index()
            imd_pass.columns=["imd_band","pass_rate"]
            fig4=px.bar(imd_pass,x="imd_band",y="pass_rate",
                        title="Pass Rate by IMD Band (Deprivation Index)",
                        template=TPL,color="pass_rate",color_continuous_scale="RdYlGn")
            fig4.update_xaxes(tickangle=40)
            st.plotly_chart(fig_bg(fig4), use_container_width=True)

    # Sunburst
    fig5=px.sunburst(fdf,path=["gender","age_band","final_result"],
                     title="Demographics Sunburst",template=TPL,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bg(fig5), use_container_width=True)

# ════════ TAB 3 — ENGAGEMENT ══════════════════════════════════
with t3:
    st.markdown('<p class="sec">📈 VLE Engagement Analysis</p>', unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        order=["Distinction","Pass","Fail","Withdrawn"]
        order_p=[o for o in order if o in fdf["final_result"].unique()]
        clean=fdf[fdf["total_clicks"]<fdf["total_clicks"].quantile(0.98)]
        fig=px.box(clean,x="final_result",y="total_clicks",
                   category_orders={"final_result":order_p},
                   title="Total VLE Clicks vs Result",template=TPL,color="final_result",
                   color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                        "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig), use_container_width=True)

    with c2:
        fig2=px.scatter(fdf.sample(min(2000,len(fdf))),
                        x="total_clicks",y="avg_assessment_score",
                        color="final_result",opacity=0.6,
                        title="Clicks vs Assessment Score",template=TPL,
                        color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                             "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig2), use_container_width=True)

    if vle_data is not None:
        st.markdown('<p class="sec">VLE Daily Activity Time Series</p>', unsafe_allow_html=True)
        daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
        daily.columns=["day","clicks"]
        fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                     title="Total Clicks Per Day Across All Students",
                     template=TPL,color_discrete_sequence=["#4f8ef7"])
        fig3.add_vline(x=30, line_dash="dash",line_color="orange",
                       annotation_text="Day 30",annotation_position="top right")
        fig3.add_vline(x=100,line_dash="dash",line_color="red",
                       annotation_text="Day 100",annotation_position="top right")
        st.plotly_chart(fig_bg(fig3), use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        if "active_days" in fdf.columns:
            fig4=px.histogram(fdf,x="active_days",color="final_result",
                              nbins=40,barmode="overlay",opacity=0.6,
                              title="Active Days Distribution",template=TPL,
                              color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                                   "Distinction":"#3498db","Withdrawn":"#95a5a6"})
            st.plotly_chart(fig_bg(fig4), use_container_width=True)
    with c4:
        if "clicks_day30" in fdf.columns:
            fig5=px.box(fdf,x="final_result",y="clicks_day30",
                        title="Clicks by Day 30 vs Result (Early Warning)",
                        template=TPL,color="final_result",
                        color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                             "Distinction":"#3498db","Withdrawn":"#95a5a6"})
            st.plotly_chart(fig_bg(fig5), use_container_width=True)

# ════════ TAB 4 — ASSESSMENTS ════════════════════════════════
with t4:
    st.markdown('<p class="sec">📝 Assessment Performance</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)

    with c1:
        fig=px.histogram(fdf,x="avg_assessment_score",color="final_result",
                         nbins=40,barmode="overlay",opacity=0.65,
                         title="Assessment Score Distribution",template=TPL,
                         color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                              "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig), use_container_width=True)

    with c2:
        score_by_res=fdf.groupby("final_result")["avg_assessment_score"].mean().reset_index()
        fig2=px.bar(score_by_res,x="final_result",y="avg_assessment_score",
                    color="final_result",title="Avg Score by Result",
                    template=TPL,text_auto=".1f",
                    color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                         "Distinction":"#3498db","Withdrawn":"#95a5a6"})
        st.plotly_chart(fig_bg(fig2), use_container_width=True)

    if "tma_avg_score" in fdf.columns and "exam_avg_score" in fdf.columns:
        c3,c4=st.columns(2)
        with c3:
            fig3=px.scatter(fdf.sample(min(2000,len(fdf))),
                            x="tma_avg_score",y="exam_avg_score",
                            color="final_result",opacity=0.6,
                            title="TMA Score vs Exam Score",template=TPL,
                            color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c",
                                                 "Distinction":"#3498db","Withdrawn":"#95a5a6"})
            st.plotly_chart(fig_bg(fig3), use_container_width=True)
        with c4:
            if "late_submissions" in fdf.columns:
                late_pass=fdf.groupby(pd.cut(fdf["late_submissions"],[0,1,3,5,20]))["target"].mean().mul(100).reset_index()
                late_pass.columns=["late_range","pass_rate"]
                late_pass["late_range"]=late_pass["late_range"].astype(str)
                fig4=px.bar(late_pass,x="late_range",y="pass_rate",
                            title="Late Submissions vs Pass Rate",template=TPL,
                            color="pass_rate",color_continuous_scale="RdYlGn")
                st.plotly_chart(fig_bg(fig4), use_container_width=True)

# ════════ TAB 5 — PREDICT ════════════════════════════════════
with t5:
    st.markdown('<p class="sec">🤖 Student Risk Predictor</p>', unsafe_allow_html=True)

    if model is None or feature_cols is None:
        st.warning("Run 03_ml_models.py first to train and save a model.")
    else:
        # Model summary
        model_path = os.path.join(OUTPUT_DIR,"03_model_summary.csv")
        if os.path.exists(model_path):
            df_sum = pd.read_csv(model_path,index_col=0)
            st.dataframe(df_sum.style.highlight_max(axis=0,color="#2ecc7144"), use_container_width=True)

        st.markdown("---")
        st.markdown("### Enter Student Details")

        col1,col2,col3=st.columns(3)
        with col1:
            gender     = st.selectbox("Gender", master["gender"].unique())
            region     = st.selectbox("Region", master["region"].unique())
            education  = st.selectbox("Education", master["highest_education"].unique())
        with col2:
            age_band   = st.selectbox("Age Band", master["age_band"].unique())
            imd_band   = st.selectbox("IMD Band", master["imd_band"].dropna().unique())
            disability = st.selectbox("Disability", master["disability"].unique())
        with col3:
            studied_credits = st.slider("Studied Credits", 0, 300, 60, step=30)
            prev_attempts   = st.slider("Prev Attempts", 0, 5, 0)
            clicks_30_val   = st.slider("Clicks by Day 30", 0, 2000, 200)
            avg_score_val   = st.slider("Avg Assessment Score", 0, 100, 60)
            active_days_val = st.slider("Active Days", 0, 200, 50)

        if st.button("🔮 Predict Pass/Fail Risk", type="primary", use_container_width=True):
            # Build input row matching feature_cols
            row = {c: 0 for c in feature_cols}

            enc_map = {
                "gender_enc":     master["gender"].unique().tolist().index(gender) if gender in master["gender"].unique() else 0,
                "region_enc":     master["region"].unique().tolist().index(region) if region in master["region"].unique() else 0,
                "highest_education_enc": master["highest_education"].unique().tolist().index(education) if education in master["highest_education"].unique() else 0,
                "imd_band_enc":   master["imd_band"].dropna().unique().tolist().index(imd_band) if imd_band in master["imd_band"].dropna().unique() else 0,
                "age_band_enc":   master["age_band"].unique().tolist().index(age_band) if age_band in master["age_band"].unique() else 0,
                "disability_enc": 0 if disability=="N" else 1,
            }
            for k,v in enc_map.items():
                if k in row: row[k]=v

            row.update({
                "studied_credits":       studied_credits,
                "num_of_prev_attempts":  prev_attempts,
                "clicks_day30":          clicks_30_val,
                "avg_assessment_score":  avg_score_val,
                "active_days":           active_days_val,
                "total_clicks":          clicks_30_val * 3,
            })

            X_input = pd.DataFrame([row])[feature_cols]
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]

            st.markdown("---")
            if pred==1:
                st.success(f"✅ **LIKELY TO PASS** — Confidence: {prob[1]*100:.1f}%")
            else:
                st.error(f"⚠️ **AT RISK (Fail/Withdraw)** — Risk Score: {prob[0]*100:.1f}%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1]*100,
                title={"text":"Pass Probability (%)"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#2ecc71" if pred==1 else "#e74c3c"},
                       "steps":[{"range":[0,40],"color":"#2d1a1a"},
                                 {"range":[40,60],"color":"#2d2a1a"},
                                 {"range":[60,100],"color":"#1a2d1a"}],
                       "threshold":{"line":{"color":"white","width":3},"value":50}}
            ))
            gauge.update_layout(template=TPL, paper_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(gauge, use_container_width=True)

st.markdown("---")
st.caption("🎓 OULAD Analytics Dashboard | Python + Streamlit | Open University Data")
