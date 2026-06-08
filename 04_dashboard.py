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

OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"
DATA_DIR    = "data"
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
.model-card{background:linear-gradient(135deg,#1a1f35,#232840);
            border-radius:12px;padding:14px;
            border:1px solid #2d3555;margin:4px}
.model-card h4{color:#4f8ef7;margin:0 0 4px}
.model-card p{color:#8892b0;font-size:12px;margin:0;line-height:1.6}
</style>""", unsafe_allow_html=True)


# ─── DATA LOADERS ──────────────────────────────────────────────
@st.cache_data
def load_master():
    for folder in ["sample_data","outputs","data"]:
        p = os.path.join(folder,"master_dataset.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_data
def load_vle():
    for folder in ["sample_data","data"]:
        p = os.path.join(folder,"studentVle.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# ─── TRAIN ALL 4 MODELS ────────────────────────────────────────
@st.cache_resource
def train_all_models(master):
    """
    Train all 4 models fresh.
    IMPORTANT: saves category_maps so prediction uses
    EXACTLY the same encoding as training — fixes AT RISK bug.
    """
    from sklearn.linear_model    import LogisticRegression
    from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.impute          import SimpleImputer
    from sklearn.preprocessing   import StandardScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import accuracy_score, roc_auc_score

    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

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

    ml_df = master.copy()

    # ── Save EXACT category → code mapping used in training ──────
    # This is the KEY FIX — prediction must use same map!
    category_maps = {}
    for col in cat_cols:
        if col in ml_df.columns:
            cat = pd.Categorical(ml_df[col])
            # map: "Male" → 0, "Female" → 1  (exact same as training)
            category_maps[col] = {v: i for i, v in enumerate(cat.categories)}
            ml_df[col+"_enc"] = cat.codes

    enc_features = [c+"_enc" for c in cat_cols if c in master.columns]
    feature_cols = enc_features + [f for f in num_features if f in ml_df.columns]
    feature_cols = [f for f in feature_cols if f in ml_df.columns]

    if "target" not in ml_df.columns:
        ml_df["target"] = ml_df["final_result"].map(PASS_LABELS)

    X = ml_df[feature_cols].fillna(0)
    y = ml_df["target"].fillna(0).astype(int)

    # Show class balance for debug
    pass_count = int(y.sum())
    fail_count = int((y==0).sum())

    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    definitions = {
        "🔵 Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, random_state=42,
                                            class_weight="balanced"))
        ]),
        "🌲 Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf",     RandomForestClassifier(
                            n_estimators=100, max_depth=12,
                            random_state=42, n_jobs=-1,
                            class_weight="balanced"))
        ]),
        "📈 Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf",     GradientBoostingClassifier(
                            n_estimators=100, max_depth=5,
                            learning_rate=0.05, random_state=42))
        ]),
    }
    if HAS_XGB:
        ratio = fail_count / max(pass_count, 1)
        definitions["⚡ XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf",     XGBClassifier(
                            n_estimators=100, max_depth=6,
                            learning_rate=0.05, random_state=42,
                            scale_pos_weight=ratio,
                            eval_metric="logloss", verbosity=0))
        ])

    trained = {}
    for name, pipe in definitions.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        acc    = round(accuracy_score(y_test, y_pred)*100, 1)
        auc    = round(roc_auc_score(y_test, y_prob), 4)
        trained[name] = {
            "pipe":   pipe,
            "acc":    acc,
            "auc":    auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": y_test,
        }

    return trained, feature_cols, category_maps


# ─── LOAD DATA ─────────────────────────────────────────────────
master   = load_master()
vle_data = load_vle()

if master is None:
    st.error("❌ No data found. Run pipeline first (python run_all.py).")
    st.stop()

if "target" not in master.columns:
    master["target"] = master["final_result"].map(PASS_LABELS)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    genders = st.multiselect("Gender",
                  sorted(master["gender"].dropna().unique()),
                  default=list(master["gender"].dropna().unique()))
    ages    = st.multiselect("Age Band",
                  sorted(master["age_band"].dropna().unique()),
                  default=list(master["age_band"].dropna().unique()))
    edus    = st.multiselect("Education",
                  master["highest_education"].dropna().unique(),
                  default=list(master["highest_education"].dropna().unique()))
    results = st.multiselect("Final Result",
                  master["final_result"].dropna().unique(),
                  default=list(master["final_result"].dropna().unique()))
    st.markdown("---")
    score_range = st.slider("Avg Score Range", 0, 100, (0,100)) \
        if "avg_assessment_score" in master.columns else (0,100)
    st.markdown("---")
    st.download_button("⬇️ Download CSV",
                        master.to_csv(index=False),"data.csv","text/csv")

fdf = master[
    master["gender"].isin(genders) &
    master["age_band"].isin(ages) &
    master["highest_education"].isin(edus) &
    master["final_result"].isin(results)
].copy()
if "avg_assessment_score" in fdf.columns:
    fdf = fdf[fdf["avg_assessment_score"].between(score_range[0],score_range[1])]

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("# 🎓 Open University Learning Analytics")
st.caption(f"Showing **{len(fdf):,}** of **{len(master):,}** students")

# ─── KPIs ──────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
pass_rate  = (fdf["target"]==1).mean()*100       if "target" in fdf.columns else 0
avg_clicks = fdf["total_clicks"].mean()           if "total_clicks" in fdf.columns else 0
avg_score  = fdf["avg_assessment_score"].mean()   if "avg_assessment_score" in fdf.columns else 0
act_days   = fdf["active_days"].mean()            if "active_days" in fdf.columns else 0
withdrew   = fdf["withdrew"].mean()*100           if "withdrew" in fdf.columns else 0

for (label,val,col,clr) in [
    ("Total Students", f"{len(fdf):,}",     k1,"#4f8ef7"),
    ("Pass Rate",      f"{pass_rate:.1f}%", k2,"#00d9a3"),
    ("Avg VLE Clicks", f"{avg_clicks:,.0f}",k3,"#ff6b9d"),
    ("Avg Score",      f"{avg_score:.1f}",  k4,"#ffb347"),
    ("Active Days",    f"{act_days:.0f}",   k5,"#a78bfa"),
    ("Withdrawal",     f"{withdrew:.1f}%",  k6,"#38bdf8"),
]:
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
CLR = {"Pass":"#2ecc71","Distinction":"#3498db",
       "Fail":"#e74c3c","Withdrawn":"#95a5a6"}

def cf(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ══════ TAB 1 — OVERVIEW ══════════════════════════════════════
with t1:
    st.markdown('<p class="sec">📊 Result Distribution</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        rc=fdf["final_result"].value_counts().reset_index()
        rc.columns=["result","count"]
        fig=px.bar(rc,x="result",y="count",color="result",
                   color_discrete_map=CLR,template=TPL,
                   title="Final Result Distribution",text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(cf(fig), use_container_width=True)
    with c2:
        if "code_module" in fdf.columns:
            mp=fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); mp.columns=["module","pass_rate"]
            fig2=px.bar(mp.sort_values("pass_rate",ascending=True),
                        x="pass_rate",y="module",orientation="h",
                        title="Pass Rate by Module",template=TPL,
                        color="pass_rate",color_continuous_scale="Tealgrn")
            st.plotly_chart(cf(fig2), use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        fig3=px.pie(fdf["final_result"].value_counts().reset_index(),
                    names="final_result",values="count",
                    color="final_result",color_discrete_map=CLR,
                    title="Result Share",template=TPL)
        st.plotly_chart(cf(fig3), use_container_width=True)
    with c4:
        if "code_presentation" in fdf.columns:
            pp=fdf.groupby("code_presentation").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); pp.columns=["presentation","pass_rate"]
            fig4=px.line(pp,x="presentation",y="pass_rate",markers=True,
                         title="Pass Rate by Presentation",template=TPL)
            st.plotly_chart(cf(fig4), use_container_width=True)

# ══════ TAB 2 — DEMOGRAPHICS ══════════════════════════════════
with t2:
    st.markdown('<p class="sec">🔍 Demographic Breakdown</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        dg=pd.crosstab(fdf["gender"],fdf["final_result"]).reset_index().melt("gender")
        fig=px.bar(dg,x="gender",y="value",color="final_result",
                   barmode="group",title="Gender vs Result",
                   template=TPL,color_discrete_map=CLR)
        st.plotly_chart(cf(fig), use_container_width=True)
    with c2:
        da=pd.crosstab(fdf["age_band"],fdf["final_result"]).reset_index().melt("age_band")
        fig2=px.bar(da,x="age_band",y="value",color="final_result",
                    barmode="stack",title="Age Band vs Result",
                    template=TPL,color_discrete_map=CLR)
        st.plotly_chart(cf(fig2), use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        de=fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        de.columns=["education","pass_rate"]
        fig3=px.bar(de.sort_values("pass_rate",ascending=True),
                    x="pass_rate",y="education",orientation="h",
                    title="Pass Rate by Education",template=TPL,
                    color="pass_rate",color_continuous_scale="Blues")
        st.plotly_chart(cf(fig3), use_container_width=True)
    with c4:
        if "imd_band" in fdf.columns:
            di=fdf.groupby("imd_band")["target"].mean().mul(100).reset_index()
            di.columns=["imd_band","pass_rate"]
            fig4=px.bar(di,x="imd_band",y="pass_rate",
                        title="Pass Rate by IMD Band",template=TPL,
                        color="pass_rate",color_continuous_scale="RdYlGn")
            fig4.update_xaxes(tickangle=40)
            st.plotly_chart(cf(fig4), use_container_width=True)
    fig5=px.sunburst(fdf,path=["gender","age_band","final_result"],
                     title="Demographics Sunburst",template=TPL,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(cf(fig5), use_container_width=True)

# ══════ TAB 3 — ENGAGEMENT ════════════════════════════════════
with t3:
    st.markdown('<p class="sec">📈 VLE Engagement</p>', unsafe_allow_html=True)
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
            st.plotly_chart(cf(fig), use_container_width=True)
        with c2:
            if "avg_assessment_score" in fdf.columns:
                fig2=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="total_clicks",y="avg_assessment_score",
                                color="final_result",opacity=0.6,
                                title="Clicks vs Score",template=TPL,
                                color_discrete_map=CLR)
                st.plotly_chart(cf(fig2), use_container_width=True)
        if vle_data is not None:
            daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
            daily.columns=["day","clicks"]
            fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                         title="Total Clicks Per Day",template=TPL,
                         color_discrete_sequence=["#4f8ef7"])
            fig3.add_vline(x=30,line_dash="dash",line_color="orange",annotation_text="Day 30")
            fig3.add_vline(x=100,line_dash="dash",line_color="red",annotation_text="Day 100")
            st.plotly_chart(cf(fig3), use_container_width=True)
        c3,c4=st.columns(2)
        with c3:
            if "active_days" in fdf.columns:
                fig4=px.histogram(fdf,x="active_days",color="final_result",
                                  nbins=40,barmode="overlay",opacity=0.6,
                                  title="Active Days Distribution",
                                  template=TPL,color_discrete_map=CLR)
                st.plotly_chart(cf(fig4), use_container_width=True)
        with c4:
            if "clicks_day30" in fdf.columns:
                fig5=px.box(fdf,x="final_result",y="clicks_day30",
                            title="Clicks Day 30 — Early Warning",
                            template=TPL,color="final_result",color_discrete_map=CLR)
                st.plotly_chart(cf(fig5), use_container_width=True)

# ══════ TAB 4 — ASSESSMENTS ═══════════════════════════════════
with t4:
    st.markdown('<p class="sec">📝 Assessment Performance</p>', unsafe_allow_html=True)
    if "avg_assessment_score" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(fdf,x="avg_assessment_score",color="final_result",
                             nbins=40,barmode="overlay",opacity=0.65,
                             title="Score Distribution",template=TPL,color_discrete_map=CLR)
            st.plotly_chart(cf(fig), use_container_width=True)
        with c2:
            sr=fdf.groupby("final_result")["avg_assessment_score"].mean().reset_index()
            fig2=px.bar(sr,x="final_result",y="avg_assessment_score",
                        color="final_result",title="Avg Score by Result",
                        template=TPL,text_auto=".1f",color_discrete_map=CLR)
            st.plotly_chart(cf(fig2), use_container_width=True)
        if "tma_avg_score" in fdf.columns and "exam_avg_score" in fdf.columns:
            c3,c4=st.columns(2)
            with c3:
                fig3=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="tma_avg_score",y="exam_avg_score",
                                color="final_result",opacity=0.6,
                                title="TMA vs Exam Score",template=TPL,color_discrete_map=CLR)
                st.plotly_chart(cf(fig3), use_container_width=True)
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
                    st.plotly_chart(cf(fig4), use_container_width=True)

# ══════ TAB 5 — PREDICT (4 MODELS) ═══════════════════════════
with t5:
    st.markdown('<p class="sec">🤖 Predict with Any Model</p>', unsafe_allow_html=True)

    with st.spinner("⏳ Training all 4 models — please wait..."):
        trained_models, feature_cols, category_maps = train_all_models(master)

    # ── Model comparison table ────────────────────────────────
    st.markdown("### 📊 Model Comparison")
    comp = pd.DataFrame({
        name: {"Accuracy":f"{v['acc']}%","ROC-AUC":f"{v['auc']}"}
        for name,v in trained_models.items()
    }).T
    comp.index.name = "Model"
    st.dataframe(comp.style.highlight_max(axis=0, color="#1a3a1a"),
                 use_container_width=True)

    # ── Model info cards ──────────────────────────────────────
    st.markdown("### 🧠 Model Descriptions")
    mc = st.columns(len(trained_models))
    model_info = {
        "🔵 Logistic Regression": ("Linear separator","Fast & interpretable.\nGood baseline for comparison.","#4f8ef7"),
        "🌲 Random Forest":       ("200 trees vote","Parallel trees. Robust\nagainst overfitting.","#2ecc71"),
        "📈 Gradient Boosting":   ("Sequential correction","Each tree fixes previous\nerrors. High accuracy.","#ffb347"),
        "⚡ XGBoost":             ("Optimized boosting","Best for tabular data.\nIndustry standard.","#a78bfa"),
    }
    for i,(name,v) in enumerate(trained_models.items()):
        algo, desc, clr = model_info.get(name,("ML Model","","#4f8ef7"))
        mc[i].markdown(f"""<div class="model-card" style="border-left:3px solid {clr}">
        <h4>{name}</h4>
        <p><b>How:</b> {algo}</p>
        <p>{desc}</p>
        <p style='color:{clr};font-size:13px;margin-top:6px'>
            Acc: {v['acc']}% | AUC: {v['auc']}</p>
        </div>""", unsafe_allow_html=True)

    # ── ROC curve comparison ───────────────────────────────────
    st.markdown("### 📈 ROC Curve Comparison")
    from sklearn.metrics import roc_curve
    fig_roc = go.Figure()
    roc_colors = ["#4f8ef7","#2ecc71","#ffb347","#a78bfa"]
    for i,(name,v) in enumerate(trained_models.items()):
        fpr,tpr,_ = roc_curve(v["y_test"], v["y_prob"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={v['auc']})",
            line=dict(color=roc_colors[i%4], width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1],y=[0,1],name="Random",
        line=dict(color="grey",dash="dash")
    ))
    fig_roc.update_layout(
        title="ROC Curves — All 4 Models",template=TPL,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── STUDENT INPUT FORM ────────────────────────────────────
    st.markdown("### 🔮 Predict a Student")

    # Model selector — RIGHT HERE on the page
    selected_model = st.selectbox(
        "🧠 Choose Model for Prediction:",
        list(trained_models.keys()),
        index=len(trained_models)-1,   # Default = last = XGBoost
        help="Select which ML algorithm to use for prediction"
    )

    # Show selected model stats
    sel = trained_models[selected_model]
    sa,sb,sc = st.columns(3)
    sa.metric("Selected Model", selected_model.split(" ",1)[-1])
    sb.metric("Accuracy",       f"{sel['acc']}%")
    sc.metric("ROC-AUC",        f"{sel['auc']}")

    st.markdown("---")

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
        credits  = st.slider("Studied Credits",      0, 300, 60,  30)
        prev_att = st.slider("Prev Attempts",         0, 5,   0)
        clicks30 = st.slider("Clicks by Day 30",      0, 2000, 200)
        avg_sc   = st.slider("Avg Assessment Score",  0, 100, 60)
        act_d    = st.slider("Active Days",           0, 200, 50)

    if st.button("🔮 Predict Now", type="primary", use_container_width=True):
        try:
            # ── Build input row using SAME encoding as training ─
            row = {c: 0 for c in feature_cols}

            # Use category_maps from training — EXACT same encoding!
            cat_inputs = {
                "gender":              gender,
                "region":              region,
                "highest_education":   education,
                "imd_band":            imd_band,
                "age_band":            age_band,
                "disability":          disability,
            }
            for col, val in cat_inputs.items():
                enc_key = col + "_enc"
                if enc_key in row and col in category_maps:
                    row[enc_key] = category_maps[col].get(val, 0)

            # Also add code_module / code_presentation if in features
            if "code_module_enc" in row and "code_module" in category_maps:
                row["code_module_enc"] = 0
            if "code_presentation_enc" in row and "code_presentation" in category_maps:
                row["code_presentation_enc"] = 0

            # Numeric features — use realistic values
            row.update({
                "studied_credits":       credits,
                "num_of_prev_attempts":  prev_att,
                "clicks_day30":          clicks30,
                "avg_assessment_score":  avg_sc,
                "active_days":           act_d,
                "total_clicks":          max(clicks30 * 4, clicks30 + 100),
                "tma_avg_score":         avg_sc,
                "exam_avg_score":        avg_sc * 0.9,
                "late_submissions":      0,
                "num_modules_registered":1,
                "click_std":             clicks30 * 0.3,
                "unique_activity_types": 5,
                "last_active_day":       act_d,
            })
            X_in = pd.DataFrame([row])[feature_cols]

            # ── Run all 4 models ───────────────────────────────
            all_results = {}
            for mname, mv in trained_models.items():
                mpred = mv["pipe"].predict(X_in)[0]
                mprob = mv["pipe"].predict_proba(X_in)[0]
                all_results[mname] = {
                    "pred":      mpred,
                    "pass_pct":  round(mprob[1]*100, 1),
                    "risk_pct":  round(mprob[0]*100, 1),
                }

            st.markdown("---")

            # ══════════════════════════════════════════════════
            # BIG NUMBER CARDS — one per model
            # ══════════════════════════════════════════════════
            st.markdown("### 🎯 Passing Probability — All 4 Models")

            model_colors = {
                "🔵 Logistic Regression": "#4f8ef7",
                "🌲 Random Forest":       "#2ecc71",
                "📈 Gradient Boosting":   "#ffb347",
                "⚡ XGBoost":             "#a78bfa",
            }

            cols = st.columns(len(all_results))
            for i, (mname, res) in enumerate(all_results.items()):
                clr   = model_colors.get(mname, "#4f8ef7")
                emoji = "✅" if res["pred"]==1 else "⚠️"
                label = "PASS"  if res["pred"]==1 else "AT RISK"
                bg    = "#0d2b0d" if res["pred"]==1 else "#2b0d0d"
                cols[i].markdown(f"""
<div style="background:{bg};border-radius:14px;padding:20px 10px;
            text-align:center;border:2px solid {clr};margin:4px">
  <p style="color:{clr};font-size:12px;margin:0;font-weight:600">
    {mname}</p>
  <p style="color:#fff;font-size:42px;font-weight:800;margin:8px 0 0">
    {res['pass_pct']}%</p>
  <p style="color:#aaa;font-size:11px;margin:0">Pass Probability</p>
  <p style="color:{clr};font-size:16px;font-weight:700;margin:8px 0 0">
    {emoji} {label}</p>
  <p style="color:#888;font-size:11px;margin:4px 0 0">
    Risk: {res['risk_pct']}%</p>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Highlighted selected model result ──────────────
            sel_res = all_results[selected_model]
            st.markdown(f"#### Selected Model: **{selected_model}**")
            if sel_res["pred"]==1:
                st.success(
                    f"✅ **LIKELY TO PASS**  |  "
                    f"Pass Probability: **{sel_res['pass_pct']}%**  |  "
                    f"Risk: **{sel_res['risk_pct']}%**"
                )
            else:
                st.error(
                    f"⚠️ **AT RISK**  |  "
                    f"Pass Probability: **{sel_res['pass_pct']}%**  |  "
                    f"Risk: **{sel_res['risk_pct']}%**"
                )

            # ── Progress bars — very clear visual ─────────────
            st.markdown("#### 📊 Pass Probability — Visual Bars")
            for mname, res in all_results.items():
                clr = model_colors.get(mname,"#4f8ef7")
                is_selected = "⭐ " if mname==selected_model else "   "
                st.markdown(
                    f"**{is_selected}{mname}** — "
                    f"<span style='color:#2ecc71;font-size:18px;font-weight:700'>"
                    f"{res['pass_pct']}% pass</span> | "
                    f"<span style='color:#e74c3c;font-size:18px;font-weight:700'>"
                    f"{res['risk_pct']}% risk</span>",
                    unsafe_allow_html=True
                )
                st.progress(int(res["pass_pct"]))

            # ── Gauge for selected model ────────────────────────
            st.markdown(f"#### 🎯 Gauge — {selected_model}")
            gauge = go.Figure(go.Indicator(
                mode    = "gauge+number+delta",
                value   = sel_res["pass_pct"],
                delta   = {"reference": 50,
                           "increasing":{"color":"#2ecc71"},
                           "decreasing":{"color":"#e74c3c"}},
                number  = {"suffix":"%","font":{"size":48}},
                title   = {"text": f"Pass Probability<br><span style='font-size:14px'>"
                                   f"{selected_model}</span>"},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":1},
                    "bar":{"color":"#2ecc71" if sel_res["pred"]==1 else "#e74c3c",
                           "thickness":0.3},
                    "bgcolor":"rgba(0,0,0,0)",
                    "borderwidth":0,
                    "steps":[
                        {"range":[0,40], "color":"#2d1a1a"},
                        {"range":[40,60],"color":"#2d2a1a"},
                        {"range":[60,100],"color":"#1a2d1a"},
                    ],
                    "threshold":{
                        "line":{"color":"white","width":4},
                        "thickness":0.8,
                        "value":50
                    }
                }
            ))
            gauge.update_layout(
                template=TPL,
                paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                font={"color":"white"}
            )
            st.plotly_chart(gauge, use_container_width=True)

            # ── Bar chart comparison ────────────────────────────
            st.markdown("#### 📈 Side-by-Side Bar Chart")
            bar_df = pd.DataFrame([
                {"Model": n.split(" ",1)[-1],
                 "Pass %": r["pass_pct"],
                 "Risk %": r["risk_pct"]}
                for n,r in all_results.items()
            ])
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Pass Probability %",
                x=bar_df["Model"],
                y=bar_df["Pass %"],
                marker_color=["#2ecc71" if v>=50 else "#e74c3c"
                              for v in bar_df["Pass %"]],
                text=[f"{v}%" for v in bar_df["Pass %"]],
                textposition="outside",
                textfont=dict(size=16, color="white")
            ))
            fig_bar.add_hline(y=50, line_dash="dash",
                              line_color="white", opacity=0.5,
                              annotation_text="50% threshold")
            fig_bar.update_layout(
                title="Pass Probability by Model (%)",
                template=TPL,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,105]),
                showlegend=False,
                font=dict(size=13)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Clean summary table ────────────────────────────
            st.markdown("#### 📋 Full Results Table")
            tbl = pd.DataFrame([
                {
                    "Model":           mname,
                    "Verdict":         "✅ Pass" if r["pred"]==1 else "⚠️ At Risk",
                    "Pass Probability":f"{r['pass_pct']}%",
                    "Risk Score":      f"{r['risk_pct']}%",
                    "Selected":        "⭐ Yes" if mname==selected_model else "",
                }
                for mname,r in all_results.items()
            ]).set_index("Model")
            st.dataframe(tbl, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

st.markdown("---")
st.caption("🎓 OULAD Analytics | Python + Streamlit | Open University Dataset")
