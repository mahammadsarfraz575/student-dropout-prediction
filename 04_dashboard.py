"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 4 — Streamlit Dashboard                               ║
║   Command: streamlit run 04_dashboard.py                     ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
DATA_DIR   = "data"
PASS_LABELS = {"Pass":1,"Distinction":1,"Fail":0,"Withdrawn":0}

st.set_page_config(
    page_title="OULAD Student Analytics",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0e1117}
.kpi{background:linear-gradient(135deg,#1a1f35,#232840);
     border-radius:12px;padding:18px 12px;text-align:center;
     border-left:4px solid #4f8ef7;margin:4px}
.kpi .label{color:#8892b0;font-size:12px;margin:0}
.kpi .value{color:#fff;font-size:28px;font-weight:700;margin:4px 0}
.sec{font-size:20px;font-weight:700;color:#4f8ef7;
     border-bottom:2px solid #4f8ef7;padding-bottom:4px;margin:16px 0 12px}
.ibox{background:#1a1f35;border-radius:10px;padding:16px;
      border:1px solid #2d3555;color:#cdd6f4;font-size:14px;line-height:1.8}
</style>""", unsafe_allow_html=True)


# ─── DATA LOADERS ──────────────────────────────────────────────
@st.cache_data
def load_master():
    # Try multiple locations
    for folder in ["sample_data", "outputs", "data"]:
        p = os.path.join(folder, "master_dataset.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_data
def load_vle():
    for folder in ["sample_data", "data"]:
        p = os.path.join(folder, "studentVle.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

# ── RETRAIN MODEL FRESH (avoids version mismatch!) ─────────────
@st.cache_resource
def get_model(master):
    """
    Retrain a fresh model from master_dataset.
    This avoids sklearn version mismatch from saved .pkl files.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # Feature columns
        cat_cols = ["gender","region","highest_education",
                    "imd_band","age_band","disability",
                    "code_module","code_presentation"]
        num_features = [
            "total_clicks","active_days","clicks_day30",
            "avg_assessment_score","studied_credits",
            "num_of_prev_attempts","tma_avg_score",
            "exam_avg_score","late_submissions",
            "num_modules_registered","click_std",
            "unique_activity_types","last_active_day"
        ]

        # Encode categoricals
        ml_df = master.copy()
        for col in cat_cols:
            if col in ml_df.columns:
                ml_df[col+"_enc"] = pd.Categorical(ml_df[col]).codes

        enc_features = [c+"_enc" for c in cat_cols if c in master.columns]
        features = enc_features + [f for f in num_features if f in ml_df.columns]
        features = [f for f in features if f in ml_df.columns]

        if "target" not in ml_df.columns:
            ml_df["target"] = ml_df["final_result"].map(PASS_LABELS)

        X = ml_df[features].fillna(0)
        y = ml_df["target"].fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf",     RandomForestClassifier(
                            n_estimators=100, random_state=42, n_jobs=-1))
        ])
        pipe.fit(X_train, y_train)

        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        return pipe, features, round(acc*100,1), round(auc,4)

    except Exception as e:
        return None, [], 0, 0


# ─── LOAD DATA ─────────────────────────────────────────────────
master   = load_master()
vle_data = load_vle()

if master is None:
    st.error("❌ No data found. Run the full pipeline first (python run_all.py).")
    st.info("Then commit `sample_data/` or `outputs/master_dataset.csv` to GitHub.")
    st.stop()

if "target" not in master.columns:
    master["target"] = master["final_result"].map(PASS_LABELS)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")

    genders = st.multiselect("Gender", sorted(master["gender"].dropna().unique()),
                              default=list(master["gender"].dropna().unique()))
    ages    = st.multiselect("Age Band", sorted(master["age_band"].dropna().unique()),
                              default=list(master["age_band"].dropna().unique()))
    edus    = st.multiselect("Education",
                              master["highest_education"].dropna().unique(),
                              default=list(master["highest_education"].dropna().unique()))
    results = st.multiselect("Final Result",
                              master["final_result"].dropna().unique(),
                              default=list(master["final_result"].dropna().unique()))
    st.markdown("---")
    if "avg_assessment_score" in master.columns:
        score_range = st.slider("Avg Score Range", 0, 100, (0, 100))
    else:
        score_range = (0, 100)
    st.markdown("---")
    st.download_button("⬇️ Download Filtered CSV",
                        master.to_csv(index=False), "filtered.csv", "text/csv")

fdf = master[
    master["gender"].isin(genders) &
    master["age_band"].isin(ages) &
    master["highest_education"].isin(edus) &
    master["final_result"].isin(results)
].copy()

if "avg_assessment_score" in fdf.columns:
    fdf = fdf[fdf["avg_assessment_score"].between(score_range[0], score_range[1])]

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("# 🎓 Open University Learning Analytics")
st.caption(f"Showing **{len(fdf):,}** of **{len(master):,}** students")

# ─── KPIs ──────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)

pass_rate  = (fdf["target"]==1).mean()*100   if "target" in fdf.columns else 0
avg_clicks = fdf["total_clicks"].mean()       if "total_clicks" in fdf.columns else 0
avg_score  = fdf["avg_assessment_score"].mean() if "avg_assessment_score" in fdf.columns else 0
act_days   = fdf["active_days"].mean()        if "active_days" in fdf.columns else 0
withdrew   = fdf["withdrew"].mean()*100       if "withdrew" in fdf.columns else 0

kpis=[
    ("Total Students",  f"{len(fdf):,}",       k1,"#4f8ef7"),
    ("Pass Rate",       f"{pass_rate:.1f}%",    k2,"#00d9a3"),
    ("Avg VLE Clicks",  f"{avg_clicks:,.0f}",   k3,"#ff6b9d"),
    ("Avg Score",       f"{avg_score:.1f}",      k4,"#ffb347"),
    ("Active Days",     f"{act_days:.0f}",       k5,"#a78bfa"),
    ("Withdrawal Rate", f"{withdrew:.1f}%",      k6,"#38bdf8"),
]
for label,val,col,clr in kpis:
    col.markdown(f"""<div class="kpi" style="border-left-color:{clr}">
    <p class="label">{label}</p><p class="value">{val}</p></div>""",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 Overview","🔍 Demographics",
    "📈 Engagement","📝 Assessments","🤖 Predict"
])

TPL = "plotly_dark"
def clean_fig(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    return fig

CLR = {"Pass":"#2ecc71","Distinction":"#3498db",
       "Fail":"#e74c3c","Withdrawn":"#95a5a6"}

# ══════ TAB 1 — OVERVIEW ═════════════════════════════════════
with t1:
    st.markdown('<p class="sec">📊 Result Distribution</p>',
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        rc = fdf["final_result"].value_counts().reset_index()
        rc.columns=["result","count"]
        fig=px.bar(rc,x="result",y="count",color="result",
                   color_discrete_map=CLR,template=TPL,
                   title="Final Result Distribution",text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c2:
        if "code_module" in fdf.columns:
            mp = fdf.groupby("code_module").apply(
                lambda x: (x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index()
            mp.columns=["module","pass_rate"]
            fig2=px.bar(mp.sort_values("pass_rate",ascending=True),
                        x="pass_rate",y="module",orientation="h",
                        title="Pass Rate by Module (%)",template=TPL,
                        color="pass_rate",color_continuous_scale="Tealgrn")
            st.plotly_chart(clean_fig(fig2), use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        fig3=px.pie(fdf["final_result"].value_counts().reset_index(),
                    names="final_result",values="count",
                    color="final_result",color_discrete_map=CLR,
                    title="Result Share",template=TPL)
        st.plotly_chart(clean_fig(fig3), use_container_width=True)
    with c4:
        if "code_presentation" in fdf.columns:
            pp=fdf.groupby("code_presentation").apply(
                lambda x: (x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index()
            pp.columns=["presentation","pass_rate"]
            fig4=px.line(pp,x="presentation",y="pass_rate",markers=True,
                         title="Pass Rate by Presentation",template=TPL)
            st.plotly_chart(clean_fig(fig4), use_container_width=True)

# ══════ TAB 2 — DEMOGRAPHICS ═════════════════════════════════
with t2:
    st.markdown('<p class="sec">🔍 Demographic Breakdown</p>',
                unsafe_allow_html=True)
    c1,c2=st.columns(2)

    with c1:
        dg=pd.crosstab(fdf["gender"],fdf["final_result"]).reset_index().melt("gender")
        fig=px.bar(dg,x="gender",y="value",color="final_result",
                   barmode="group",title="Gender vs Result",
                   template=TPL,color_discrete_map=CLR)
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c2:
        da=pd.crosstab(fdf["age_band"],fdf["final_result"]).reset_index().melt("age_band")
        fig2=px.bar(da,x="age_band",y="value",color="final_result",
                    barmode="stack",title="Age Band vs Result",
                    template=TPL,color_discrete_map=CLR)
        st.plotly_chart(clean_fig(fig2), use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        de=fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        de.columns=["education","pass_rate"]
        fig3=px.bar(de.sort_values("pass_rate",ascending=True),
                    x="pass_rate",y="education",orientation="h",
                    title="Pass Rate by Education (%)",template=TPL,
                    color="pass_rate",color_continuous_scale="Blues")
        st.plotly_chart(clean_fig(fig3), use_container_width=True)

    with c4:
        if "imd_band" in fdf.columns:
            di=fdf.groupby("imd_band")["target"].mean().mul(100).reset_index()
            di.columns=["imd_band","pass_rate"]
            fig4=px.bar(di,x="imd_band",y="pass_rate",
                        title="Pass Rate by IMD Band",
                        template=TPL,color="pass_rate",
                        color_continuous_scale="RdYlGn")
            fig4.update_xaxes(tickangle=40)
            st.plotly_chart(clean_fig(fig4), use_container_width=True)

    fig5=px.sunburst(fdf,path=["gender","age_band","final_result"],
                     title="Demographics Sunburst",template=TPL,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(clean_fig(fig5), use_container_width=True)

# ══════ TAB 3 — ENGAGEMENT ═══════════════════════════════════
with t3:
    st.markdown('<p class="sec">📈 VLE Engagement Analysis</p>',
                unsafe_allow_html=True)

    if "total_clicks" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            order_p=[o for o in ["Distinction","Pass","Fail","Withdrawn"]
                     if o in fdf["final_result"].unique()]
            cl=fdf[fdf["total_clicks"]<fdf["total_clicks"].quantile(0.98)]
            fig=px.box(cl,x="final_result",y="total_clicks",
                       category_orders={"final_result":order_p},
                       title="VLE Clicks vs Result",template=TPL,
                       color="final_result",color_discrete_map=CLR)
            st.plotly_chart(clean_fig(fig), use_container_width=True)

        with c2:
            if "avg_assessment_score" in fdf.columns:
                fig2=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="total_clicks",y="avg_assessment_score",
                                color="final_result",opacity=0.6,
                                title="Clicks vs Score",template=TPL,
                                color_discrete_map=CLR)
                st.plotly_chart(clean_fig(fig2), use_container_width=True)

        if vle_data is not None:
            st.markdown('<p class="sec">Daily VLE Activity</p>',
                        unsafe_allow_html=True)
            daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
            daily.columns=["day","clicks"]
            fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                         title="Total Clicks Per Day",template=TPL,
                         color_discrete_sequence=["#4f8ef7"])
            fig3.add_vline(x=30,line_dash="dash",line_color="orange",
                           annotation_text="Day 30")
            fig3.add_vline(x=100,line_dash="dash",line_color="red",
                           annotation_text="Day 100")
            st.plotly_chart(clean_fig(fig3), use_container_width=True)

        c3,c4=st.columns(2)
        with c3:
            if "active_days" in fdf.columns:
                fig4=px.histogram(fdf,x="active_days",color="final_result",
                                  nbins=40,barmode="overlay",opacity=0.6,
                                  title="Active Days Distribution",
                                  template=TPL,color_discrete_map=CLR)
                st.plotly_chart(clean_fig(fig4), use_container_width=True)
        with c4:
            if "clicks_day30" in fdf.columns:
                fig5=px.box(fdf,x="final_result",y="clicks_day30",
                            title="Clicks by Day 30 (Early Warning)",
                            template=TPL,color="final_result",
                            color_discrete_map=CLR)
                st.plotly_chart(clean_fig(fig5), use_container_width=True)

# ══════ TAB 4 — ASSESSMENTS ══════════════════════════════════
with t4:
    st.markdown('<p class="sec">📝 Assessment Performance</p>',
                unsafe_allow_html=True)

    if "avg_assessment_score" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(fdf,x="avg_assessment_score",
                             color="final_result",nbins=40,
                             barmode="overlay",opacity=0.65,
                             title="Score Distribution",
                             template=TPL,color_discrete_map=CLR)
            st.plotly_chart(clean_fig(fig), use_container_width=True)

        with c2:
            sr=fdf.groupby("final_result")["avg_assessment_score"].mean().reset_index()
            fig2=px.bar(sr,x="final_result",y="avg_assessment_score",
                        color="final_result",title="Avg Score by Result",
                        template=TPL,text_auto=".1f",color_discrete_map=CLR)
            st.plotly_chart(clean_fig(fig2), use_container_width=True)

        if "tma_avg_score" in fdf.columns and "exam_avg_score" in fdf.columns:
            c3,c4=st.columns(2)
            with c3:
                fig3=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="tma_avg_score",y="exam_avg_score",
                                color="final_result",opacity=0.6,
                                title="TMA vs Exam Score",
                                template=TPL,color_discrete_map=CLR)
                st.plotly_chart(clean_fig(fig3), use_container_width=True)
            with c4:
                if "late_submissions" in fdf.columns:
                    lp=fdf.groupby(pd.cut(
                        fdf["late_submissions"],[0,1,3,5,20])
                    )["target"].mean().mul(100).reset_index()
                    lp.columns=["late_range","pass_rate"]
                    lp["late_range"]=lp["late_range"].astype(str)
                    fig4=px.bar(lp,x="late_range",y="pass_rate",
                                title="Late Submissions vs Pass Rate",
                                template=TPL,color="pass_rate",
                                color_continuous_scale="RdYlGn")
                    st.plotly_chart(clean_fig(fig4), use_container_width=True)

# ══════ TAB 5 — PREDICT ══════════════════════════════════════
with t5:
    st.markdown('<p class="sec">🤖 Student Risk Predictor</p>',
                unsafe_allow_html=True)
    st.info("🔄 Model trains fresh on load — no version conflicts!")

    with st.spinner("Training ML model..."):
        model, feature_cols, acc, auc = get_model(master)

    if model is None:
        st.error("Model training failed. Check data.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Model", "Random Forest")
        c2.metric("Accuracy", f"{acc}%")
        c3.metric("ROC-AUC", f"{auc}")

        st.markdown("---")
        st.markdown("### 🔮 Enter Student Details")

        col1,col2,col3 = st.columns(3)
        with col1:
            gender    = st.selectbox("Gender",
                            sorted(master["gender"].dropna().unique()))
            region    = st.selectbox("Region",
                            sorted(master["region"].dropna().unique()))
            education = st.selectbox("Education",
                            master["highest_education"].dropna().unique())
        with col2:
            age_band   = st.selectbox("Age Band",
                             sorted(master["age_band"].dropna().unique()))
            imd_band   = st.selectbox("IMD Band",
                             master["imd_band"].dropna().unique())
            disability = st.selectbox("Disability",
                             master["disability"].dropna().unique())
        with col3:
            credits    = st.slider("Studied Credits",    0, 300, 60, 30)
            prev_att   = st.slider("Prev Attempts",      0, 5, 0)
            clicks30   = st.slider("Clicks by Day 30",   0, 2000, 200)
            avg_sc     = st.slider("Avg Assessment Score", 0, 100, 60)
            act_d      = st.slider("Active Days",        0, 200, 50)

        if st.button("🔮 Predict Risk", type="primary",
                     use_container_width=True):
            try:
                # Build input row
                row = {c: 0 for c in feature_cols}

                cat_enc = {
                    "gender_enc":              sorted(master["gender"].dropna().unique()).index(gender),
                    "region_enc":              sorted(master["region"].dropna().unique()).index(region),
                    "highest_education_enc":   list(master["highest_education"].dropna().unique()).index(education),
                    "imd_band_enc":            list(master["imd_band"].dropna().unique()).index(imd_band),
                    "age_band_enc":            sorted(master["age_band"].dropna().unique()).index(age_band),
                    "disability_enc":          0 if disability == "N" else 1,
                }
                for k,v in cat_enc.items():
                    if k in row: row[k] = v

                row.update({
                    "studied_credits":        credits,
                    "num_of_prev_attempts":   prev_att,
                    "clicks_day30":           clicks30,
                    "avg_assessment_score":   avg_sc,
                    "active_days":            act_d,
                    "total_clicks":           clicks30 * 3,
                })

                X_in = pd.DataFrame([row])[feature_cols]
                pred = model.predict(X_in)[0]
                prob = model.predict_proba(X_in)[0]

                st.markdown("---")
                if pred == 1:
                    st.success(f"✅ LIKELY TO PASS — Confidence: {prob[1]*100:.1f}%")
                else:
                    st.error(f"⚠️ AT RISK — Risk Score: {prob[0]*100:.1f}%")

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1]*100,
                    title={"text":"Pass Probability (%)"},
                    gauge={
                        "axis":{"range":[0,100]},
                        "bar":{"color":"#2ecc71" if pred==1 else "#e74c3c"},
                        "steps":[
                            {"range":[0,40], "color":"#2d1a1a"},
                            {"range":[40,60],"color":"#2d2a1a"},
                            {"range":[60,100],"color":"#1a2d1a"}
                        ],
                        "threshold":{
                            "line":{"color":"white","width":3},
                            "value":50
                        }
                    }
                ))
                gauge.update_layout(
                    template=TPL,
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=280
                )
                st.plotly_chart(gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("🎓 OULAD Analytics | Python + Streamlit | Open University Dataset")
