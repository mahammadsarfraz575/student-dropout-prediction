"""
OULAD Student Analytics Dashboard — Full Rewrite
Clean website-style navigation + fixed predictions
streamlit run 04_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, warnings
warnings.filterwarnings("ignore")

PASS_LABELS = {"Pass":1,"Distinction":1,"Fail":0,"Withdrawn":0}
TPL = "plotly_dark"

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG + FULL CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OULAD Analytics",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"]{background:#090d1a}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2a3a}
[data-testid="stSidebar"] *{font-family:'Segoe UI',sans-serif}

/* ── Logo area ── */
.logo-box{padding:24px 16px 8px;border-bottom:1px solid #1e2a3a;margin-bottom:8px}
.logo-title{color:#fff;font-size:18px;font-weight:700;margin:0}
.logo-sub{color:#4f8ef7;font-size:11px;margin:2px 0 0}

/* ── Nav section label ── */
.nav-label{color:#4f8ef7;font-size:10px;font-weight:700;letter-spacing:2px;
           padding:16px 16px 4px;text-transform:uppercase}

/* ── Nav item ── */
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 16px;
          border-radius:8px;margin:2px 8px;cursor:pointer;
          color:#8892b0;font-size:13px;font-weight:500;
          transition:all .15s;border:none;background:transparent;width:calc(100% - 16px)}
.nav-item:hover{background:#1a2235;color:#fff}
.nav-item.active{background:linear-gradient(90deg,#1a3a6a,#1a2a4a);
                 color:#4f8ef7;border-left:3px solid #4f8ef7}
.nav-item.sub{padding-left:36px;font-size:12px}
.nav-item.sub.active{background:#111c30;color:#7ab3f7;border-left:2px solid #4f8ef7}

/* ── KPI card ── */
.kpi{background:linear-gradient(135deg,#111827,#1a2235);
     border-radius:12px;padding:18px 14px;text-align:center;
     border-left:4px solid #4f8ef7;margin:2px;
     box-shadow:0 2px 12px rgba(0,0,0,.3)}
.kpi .lb{color:#6b7a99;font-size:11px;font-weight:600;
         letter-spacing:1px;text-transform:uppercase;margin:0}
.kpi .vl{color:#fff;font-size:30px;font-weight:800;margin:6px 0 2px;line-height:1}
.kpi .dl{font-size:11px;margin:0}

/* ── Section header ── */
.sec{font-size:22px;font-weight:700;color:#e2e8f0;margin:0 0 4px}
.sec-sub{color:#6b7a99;font-size:13px;margin:0 0 20px}

/* ── Model card ── */
.mcard{background:#111827;border-radius:12px;padding:16px;
       border:1px solid #1e2a3a;height:100%;transition:border .15s}
.mcard:hover{border-color:#4f8ef7}
.mcard .mn{font-size:13px;font-weight:700;margin:0 0 4px}
.mcard .md{color:#6b7a99;font-size:11px;margin:0 0 10px;line-height:1.5}
.mcard .ms{font-size:20px;font-weight:800;margin:0}

/* ── Result card ── */
.res-pass{background:linear-gradient(135deg,#0a2a0a,#0d3320);
          border:2px solid #2ecc71;border-radius:14px;
          padding:24px;text-align:center}
.res-risk{background:linear-gradient(135deg,#2a0a0a,#330d0d);
          border:2px solid #e74c3c;border-radius:14px;
          padding:24px;text-align:center}
.res-pct{font-size:56px;font-weight:900;margin:0;line-height:1}
.res-label{font-size:14px;font-weight:600;letter-spacing:2px;
           text-transform:uppercase;margin:6px 0 0}

/* ── Page banner ── */
.page-banner{background:linear-gradient(135deg,#0f1f3d,#0a1628);
             border-radius:16px;padding:28px 32px;margin-bottom:24px;
             border:1px solid #1e2a3a}
.page-banner h1{color:#fff;font-size:26px;font-weight:800;margin:0 0 6px}
.page-banner p{color:#6b7a99;font-size:14px;margin:0}

/* ── Table ── */
.styled-table{width:100%;border-collapse:collapse;font-size:13px}
.styled-table th{background:#111827;color:#4f8ef7;padding:10px 14px;
                 text-align:left;font-weight:600;border-bottom:2px solid #1e2a3a}
.styled-table td{padding:10px 14px;border-bottom:1px solid #1a2235;color:#e2e8f0}
.styled-table tr:hover td{background:#111827}

/* ── Progress bar ── */
.prob-row{display:flex;align-items:center;gap:12px;margin:8px 0}
.prob-label{min-width:180px;font-size:12px;color:#8892b0;font-weight:600}
.prob-bar-bg{flex:1;height:10px;background:#1a2235;border-radius:10px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:10px;transition:width .4s}
.prob-pct{min-width:48px;text-align:right;font-size:13px;font-weight:700}

/* ── Breadcrumb ── */
.breadcrumb{color:#6b7a99;font-size:12px;margin-bottom:12px}
.breadcrumb span{color:#4f8ef7}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE — tracks current page
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "home"

def nav(p):
    st.session_state.page = p


# ══════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════
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

@st.cache_resource
def train_all_models(df_hash):
    """Train 4 models. Uses df_hash for cache key only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble     import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       HistGradientBoostingClassifier)
    from sklearn.model_selection import train_test_split
    from sklearn.impute          import SimpleImputer
    from sklearn.preprocessing   import StandardScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import accuracy_score, roc_auc_score

    master = load_master()
    if master is None:
        return None, [], {}, {}

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

    ml = master.copy()
    # Exact same encoding — save map for prediction
    category_maps = {}
    for col in cat_cols:
        if col in ml.columns:
            cat = pd.Categorical(ml[col])
            category_maps[col] = dict(zip(cat.categories, range(len(cat.categories))))
            ml[col+"_enc"] = cat.codes

    enc_feats   = [c+"_enc" for c in cat_cols if c in master.columns]
    feature_cols= enc_feats + [f for f in num_features if f in ml.columns]
    feature_cols= [f for f in feature_cols if f in ml.columns]

    if "target" not in ml.columns:
        ml["target"] = ml["final_result"].map(PASS_LABELS)

    X = ml[feature_cols].fillna(0)
    y = ml["target"].fillna(0).astype(int)

    # Save feature means for smart defaults
    feature_means = X.mean().to_dict()

    X_tr,X_te,y_tr,y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    defs = {
        "🔵 Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "🌲 Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
        ]),
        "📈 Gradient Boosting": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.05, random_state=42)),
        ]),
        "⚡ Hist Gradient Boost": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=100, max_depth=6,
                learning_rate=0.05, random_state=42)),
        ]),
    }

    trained = {}
    for name, pipe in defs.items():
        pipe.fit(X_tr, y_tr)
        yp  = pipe.predict(X_te)
        ypr = pipe.predict_proba(X_te)[:,1]
        trained[name] = {
            "pipe":   pipe,
            "acc":    round(accuracy_score(y_te,yp)*100,1),
            "auc":    round(roc_auc_score(y_te,ypr),4),
            "y_pred": yp,
            "y_prob": ypr,
            "y_test": y_te,
        }

    return trained, feature_cols, category_maps, feature_means


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
master   = load_master()
vle_data = load_vle()

if master is None:
    st.error("❌ No data found. Run pipeline first → python run_all.py")
    st.stop()

if "target" not in master.columns:
    master["target"] = master["final_result"].map(PASS_LABELS)

CLR = {"Pass":"#2ecc71","Distinction":"#3498db",
       "Fail":"#e74c3c","Withdrawn":"#95a5a6"}


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-box">
      <p class="logo-title">🎓 OULAD Analytics</p>
      <p class="logo-sub">Open University Learning Intelligence</p>
    </div>""", unsafe_allow_html=True)

    pg = st.session_state.page

    # ── MAIN NAV ──────────────────────────────────────────────
    st.markdown('<p class="nav-label">Main</p>', unsafe_allow_html=True)

    if st.button("🏠  Home",
                 key="nav_home",
                 help="Dashboard overview"):
        nav("home")
    if st.button("📊  Overview",
                 key="nav_overview",
                 help="Result distribution"):
        nav("overview")

    # ── ANALYSIS SUB-MENU ─────────────────────────────────────
    st.markdown('<p class="nav-label">Analysis</p>', unsafe_allow_html=True)

    if st.button("👥  Demographics",
                 key="nav_demo",
                 help="Gender, age, education, deprivation"):
        nav("demographics")
    if st.button("📈  Engagement",
                 key="nav_engage",
                 help="VLE clicks and activity"):
        nav("engagement")
    if st.button("📝  Assessments",
                 key="nav_assess",
                 help="Scores and submissions"):
        nav("assessments")

    # ── PREDICT ───────────────────────────────────────────────
    st.markdown('<p class="nav-label">Predict</p>', unsafe_allow_html=True)

    if st.button("🤖  Single Student",
                 key="nav_predict",
                 help="Predict one student's risk"):
        nav("predict")
    if st.button("⚖️  Compare Models",
                 key="nav_compare",
                 help="Side-by-side model comparison"):
        nav("compare")

    # ── DATA ──────────────────────────────────────────────────
    st.markdown('<p class="nav-label">Data</p>', unsafe_allow_html=True)

    if st.button("🔍  Explore Data",
                 key="nav_data",
                 help="Raw data explorer"):
        nav("data")

    st.markdown("---")

    # ── GLOBAL FILTERS ────────────────────────────────────────
    st.markdown("**🎛️ Global Filters**")
    genders = st.multiselect("Gender",
                  sorted(master["gender"].dropna().unique()),
                  default=list(master["gender"].dropna().unique()),
                  label_visibility="collapsed")
    ages    = st.multiselect("Age Band",
                  sorted(master["age_band"].dropna().unique()),
                  default=list(master["age_band"].dropna().unique()),
                  label_visibility="collapsed")
    results_f = st.multiselect("Result",
                  master["final_result"].dropna().unique(),
                  default=list(master["final_result"].dropna().unique()),
                  label_visibility="collapsed")

    st.markdown("---")
    st.download_button("⬇️ Export CSV",
                        master.to_csv(index=False),
                        "oulad_data.csv","text/csv",
                        use_container_width=True)
    st.caption(f"v2.0 · {len(master):,} students")


# ── Apply filters ─────────────────────────────────────────────
fdf = master[
    master["gender"].isin(genders) &
    master["age_band"].isin(ages) &
    master["final_result"].isin(results_f)
].copy()


# ══════════════════════════════════════════════════════════════
# HELPER — KPI row
# ══════════════════════════════════════════════════════════════
def kpi_row():
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    data = [
        ("STUDENTS",    f"{len(fdf):,}",                         k1,"#4f8ef7",""),
        ("PASS RATE",   f"{(fdf['target']==1).mean()*100:.1f}%", k2,"#2ecc71",""),
        ("AVG CLICKS",  f"{fdf['total_clicks'].mean():,.0f}"
                         if "total_clicks" in fdf.columns else "N/A",
                                                                  k3,"#ff6b9d",""),
        ("AVG SCORE",   f"{fdf['avg_assessment_score'].mean():.1f}"
                         if "avg_assessment_score" in fdf.columns else "N/A",
                                                                  k4,"#ffb347",""),
        ("ACTIVE DAYS", f"{fdf['active_days'].mean():.0f}"
                         if "active_days" in fdf.columns else "N/A",
                                                                  k5,"#a78bfa",""),
        ("WITHDRAWAL",  f"{fdf['withdrew'].mean()*100:.1f}%"
                         if "withdrew" in fdf.columns else "N/A",
                                                                  k6,"#38bdf8",""),
    ]
    for label,val,col,clr,dl in data:
        col.markdown(f"""<div class="kpi" style="border-left-color:{clr}">
        <p class="lb">{label}</p>
        <p class="vl">{val}</p>
        <p class="dl" style="color:{clr}">{dl}</p>
        </div>""", unsafe_allow_html=True)

def cf(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e2e8f0")
    return fig

def banner(title, subtitle="", icon=""):
    st.markdown(f"""
    <div class="page-banner">
      <h1>{icon} {title}</h1>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    banner("OULAD Learning Analytics",
           "Predicting student dropout risk using engagement data from 32,593 students",
           "🎓")
    kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        rc = fdf["final_result"].value_counts().reset_index()
        rc.columns = ["result","count"]
        fig = px.bar(rc, x="result", y="count", color="result",
                     color_discrete_map=CLR, template=TPL,
                     title="📊 Final Result Distribution", text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(cf(fig), use_container_width=True)

    with c2:
        if "code_module" in fdf.columns:
            mp = fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index()
            mp.columns=["module","pass_rate"]
            fig2 = px.bar(mp.sort_values("pass_rate", ascending=True),
                          x="pass_rate", y="module", orientation="h",
                          title="🏫 Pass Rate by Module (%)", template=TPL,
                          color="pass_rate", color_continuous_scale="Tealgrn")
            st.plotly_chart(cf(fig2), use_container_width=True)

    # Quick navigation cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧭 Quick Navigation")
    n1,n2,n3,n4 = st.columns(4)

    cards = [
        ("📊","Overview","Result distribution\nand module analysis","overview","#4f8ef7"),
        ("👥","Demographics","Gender, age, education\nand deprivation gaps","demographics","#2ecc71"),
        ("📈","Engagement","VLE clicks and\nlearning activity","engagement","#ffb347"),
        ("🤖","Predict","Predict student\ndropout risk","predict","#a78bfa"),
    ]
    for col,(ic,title,desc,pg_key,clr) in zip([n1,n2,n3,n4], cards):
        with col:
            if st.button(f"{ic} {title}\n\n{desc}",
                         key=f"home_{pg_key}",
                         use_container_width=True):
                nav(pg_key)
            st.markdown(
                f"<div style='height:4px;background:{clr};"
                f"border-radius:2px;margin-top:-8px'></div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "overview":
    banner("Result Overview",
           f"Analysing {len(fdf):,} student outcomes across all modules", "📊")
    kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        rc = fdf["final_result"].value_counts().reset_index()
        rc.columns=["result","count"]
        fig=px.bar(rc,x="result",y="count",color="result",
                   color_discrete_map=CLR,template=TPL,
                   title="Final Result Distribution",text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(cf(fig), use_container_width=True)
    with c2:
        fig2=px.pie(rc,names="result",values="count",
                    color="result",color_discrete_map=CLR,
                    title="Result Share",template=TPL,hole=0.4)
        st.plotly_chart(cf(fig2), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        if "code_module" in fdf.columns:
            mp=fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); mp.columns=["module","pass_rate"]
            fig3=px.bar(mp.sort_values("pass_rate",ascending=True),
                        x="pass_rate",y="module",orientation="h",
                        title="Pass Rate by Module",template=TPL,
                        color="pass_rate",color_continuous_scale="Tealgrn")
            st.plotly_chart(cf(fig3), use_container_width=True)
    with c4:
        if "code_presentation" in fdf.columns:
            pp=fdf.groupby("code_presentation").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); pp.columns=["pres","pass_rate"]
            fig4=px.line(pp,x="pres",y="pass_rate",markers=True,
                         title="Pass Rate by Presentation",template=TPL)
            st.plotly_chart(cf(fig4), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "demographics":
    banner("Demographics Analysis",
           "Understanding how student background affects outcomes", "👥")
    kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
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

    c3,c4 = st.columns(2)
    with c3:
        de=fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        de.columns=["education","pass_rate"]
        fig3=px.bar(de.sort_values("pass_rate",ascending=True),
                    x="pass_rate",y="education",orientation="h",
                    title="Pass Rate by Education (%)",template=TPL,
                    color="pass_rate",color_continuous_scale="Blues")
        st.plotly_chart(cf(fig3), use_container_width=True)
    with c4:
        if "imd_band" in fdf.columns:
            di=fdf.groupby("imd_band")["target"].mean().mul(100).reset_index()
            di.columns=["imd_band","pass_rate"]
            fig4=px.bar(di,x="imd_band",y="pass_rate",
                        title="Pass Rate by Deprivation (IMD Band)",
                        template=TPL,color="pass_rate",
                        color_continuous_scale="RdYlGn")
            fig4.update_xaxes(tickangle=40)
            st.plotly_chart(cf(fig4), use_container_width=True)

    fig5=px.sunburst(fdf,path=["gender","age_band","final_result"],
                     title="Demographics Sunburst",template=TPL,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(cf(fig5), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ENGAGEMENT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "engagement":
    banner("VLE Engagement Analysis",
           "How online learning activity predicts student success", "📈")
    kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)

    if "total_clicks" in fdf.columns:
        c1,c2 = st.columns(2)
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
            st.markdown("#### 🕐 Daily VLE Activity Timeline")
            daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
            daily.columns=["day","clicks"]
            fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                         title="Total Clicks Per Day",template=TPL,
                         color_discrete_sequence=["#4f8ef7"])
            fig3.add_vline(x=30,line_dash="dash",line_color="orange",
                           annotation_text="Day 30 — Early Warning")
            fig3.add_vline(x=100,line_dash="dash",line_color="red",
                           annotation_text="Day 100")
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
                            title="Clicks by Day 30 — Early Warning Signal",
                            template=TPL,color="final_result",
                            color_discrete_map=CLR)
                st.plotly_chart(cf(fig5), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ASSESSMENTS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "assessments":
    banner("Assessment Performance",
           "Scores, submission behaviour and academic trends", "📝")
    kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)

    if "avg_assessment_score" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(fdf,x="avg_assessment_score",color="final_result",
                             nbins=40,barmode="overlay",opacity=0.65,
                             title="Score Distribution",template=TPL,
                             color_discrete_map=CLR)
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
                                title="TMA vs Exam Score",template=TPL,
                                color_discrete_map=CLR)
                st.plotly_chart(cf(fig3), use_container_width=True)
            with c4:
                if "late_submissions" in fdf.columns:
                    bins=[0,1,3,5,20]
                    lp=fdf.groupby(pd.cut(fdf["late_submissions"],bins))["target"]\
                          .mean().mul(100).reset_index()
                    lp.columns=["late_range","pass_rate"]
                    lp["late_range"]=lp["late_range"].astype(str)
                    fig4=px.bar(lp,x="late_range",y="pass_rate",
                                title="Late Submissions vs Pass Rate",
                                template=TPL,color="pass_rate",
                                color_continuous_scale="RdYlGn")
                    st.plotly_chart(cf(fig4), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT — SINGLE STUDENT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "predict":
    banner("Student Risk Predictor",
           "Enter student details to predict pass/fail probability", "🤖")

    with st.spinner("⏳ Training models..."):
        trained, feature_cols, category_maps, feature_means = \
            train_all_models(len(master))

    if trained is None:
        st.error("Model training failed.")
        st.stop()

    # ── Model selector ────────────────────────────────────────
    model_names = list(trained.keys())
    model_colors_map = {
        "🔵 Logistic Regression": "#4f8ef7",
        "🌲 Random Forest":        "#2ecc71",
        "📈 Gradient Boosting":    "#ffb347",
        "⚡ Hist Gradient Boost":  "#a78bfa",
    }

    sel_name = st.selectbox(
        "🧠 Select Model to Highlight:",
        model_names,
        index=2,
        help="All 4 models predict — this one is highlighted"
    )

    st.markdown("---")
    st.markdown("### 📋 Student Details")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**👤 Personal**")
        gender    = st.selectbox("Gender",
                        sorted(master["gender"].dropna().unique()))
        age_band  = st.selectbox("Age Band",
                        sorted(master["age_band"].dropna().unique()))
        disability= st.selectbox("Disability",
                        master["disability"].dropna().unique())
        region    = st.selectbox("Region",
                        sorted(master["region"].dropna().unique()))

    with c2:
        st.markdown("**🎓 Academic**")
        education = st.selectbox("Education Level",
                        master["highest_education"].dropna().unique())
        imd_band  = st.selectbox("IMD Band (Deprivation)",
                        master["imd_band"].dropna().unique())
        credits   = st.select_slider("Studied Credits",
                        options=[30,60,90,120,150,180,210,240,270,300],
                        value=60)
        prev_att  = st.slider("Previous Attempts", 0, 5, 0)

    with c3:
        st.markdown("**📊 Engagement**")
        avg_sc   = st.slider("Avg Assessment Score", 0, 100, 65)
        clicks30 = st.slider("Clicks by Day 30",     0, 3000, 500)
        act_d    = st.slider("Active Days",           0, 250, 80)

    st.markdown("---")

    predict_btn = st.button("🔮  Run Prediction — All 4 Models",
                             type="primary", use_container_width=True)

    if predict_btn:
        # ── Build input — use feature_means as base (NOT zeros!) ─
        row = dict(feature_means)   # start with column averages

        # Override with exact encoded values from training
        cat_inputs = {
            "gender":            gender,
            "region":            region,
            "highest_education": education,
            "imd_band":          imd_band,
            "age_band":          age_band,
            "disability":        disability,
        }
        for col, val in cat_inputs.items():
            key = col + "_enc"
            if key in row and col in category_maps:
                row[key] = category_maps[col].get(val, 0)

        # User inputs
        row.update({
            "avg_assessment_score":  avg_sc,
            "tma_avg_score":         avg_sc,
            "exam_avg_score":        max(0, avg_sc - 5),
            "clicks_day30":          clicks30,
            "total_clicks":          clicks30 * 4,
            "active_days":           act_d,
            "last_active_day":       act_d,
            "click_std":             clicks30 * 0.25,
            "unique_activity_types": 6,
            "studied_credits":       credits,
            "num_of_prev_attempts":  prev_att,
            "num_modules_registered":1,
            "late_submissions":      0,
        })

        X_in = pd.DataFrame([row])[feature_cols]

        # ── Run all 4 models ──────────────────────────────────
        results = {}
        for mname, mv in trained.items():
            pred = mv["pipe"].predict(X_in)[0]
            prob = mv["pipe"].predict_proba(X_in)[0]
            results[mname] = {
                "pred":     int(pred),
                "pass_pct": round(float(prob[1])*100, 1),
                "risk_pct": round(float(prob[0])*100, 1),
            }

        # ════════════════════════════════════════════════════
        # BIG RESULT CARDS — 4 columns
        # ════════════════════════════════════════════════════
        st.markdown("### 🎯 Prediction Results")
        cols = st.columns(4)
        for i,(mname,res) in enumerate(results.items()):
            clr   = model_colors_map.get(mname,"#4f8ef7")
            is_sel= mname == sel_name
            icon  = "✅" if res["pred"]==1 else "⚠️"
            label = "PASS"   if res["pred"]==1 else "AT RISK"
            bg    = "#0a2a0a" if res["pred"]==1 else "#2a0a0a"
            bclr  = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            star  = "⭐ " if is_sel else ""
            brd   = f"3px solid {clr}" if is_sel else f"1px solid {bclr}"

            cols[i].markdown(f"""
<div style="background:{bg};border-radius:14px;padding:22px 12px;
     text-align:center;border:{brd};margin:4px 0">
  <p style="color:{clr};font-size:11px;font-weight:700;
     letter-spacing:1px;text-transform:uppercase;margin:0">{star}{mname.split(' ',1)[-1]}</p>
  <p style="color:{bclr};font-size:52px;font-weight:900;
     margin:10px 0 0;line-height:1">{res['pass_pct']}%</p>
  <p style="color:#888;font-size:11px;margin:2px 0 8px">Pass Probability</p>
  <p style="color:{bclr};font-size:15px;font-weight:700;
     letter-spacing:1px;margin:0">{icon} {label}</p>
  <p style="color:#666;font-size:11px;margin:6px 0 0">Risk: {res['risk_pct']}%</p>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Progress bars ─────────────────────────────────────
        st.markdown("### 📊 Pass Probability Bars")
        for mname, res in results.items():
            clr  = model_colors_map.get(mname,"#4f8ef7")
            bclr = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            star = "⭐ " if mname==sel_name else "　 "
            pct  = res["pass_pct"]

            st.markdown(f"""
<div class="prob-row">
  <div class="prob-label">{star}{mname}</div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill"
         style="width:{pct}%;background:{bclr}"></div>
  </div>
  <div class="prob-pct" style="color:{bclr}">{pct}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge for selected model ───────────────────────────
        sel_res = results[sel_name]
        gcol = "#2ecc71" if sel_res["pred"]==1 else "#e74c3c"
        st.markdown(f"### 🎯 Gauge — {sel_name}")
        gauge = go.Figure(go.Indicator(
            mode    = "gauge+number+delta",
            value   = sel_res["pass_pct"],
            delta   = {"reference":50,
                       "increasing":{"color":"#2ecc71"},
                       "decreasing":{"color":"#e74c3c"}},
            number  = {"suffix":"%","font":{"size":52,"color":gcol}},
            title   = {"text":f"Pass Probability<br>"
                              f"<span style='font-size:13px;color:#6b7a99'>"
                              f"{sel_name}</span>"},
            gauge   = {
                "axis":{"range":[0,100],"tickwidth":1},
                "bar" :{"color":gcol,"thickness":0.3},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,40], "color":"#1a0a0a"},
                    {"range":[40,60],"color":"#1a1a0a"},
                    {"range":[60,100],"color":"#0a1a0a"},
                ],
                "threshold":{
                    "line":{"color":"white","width":4},
                    "thickness":0.8,"value":50
                }
            }
        ))
        gauge.update_layout(
            template=TPL, paper_bgcolor="rgba(0,0,0,0)", height=300,
            font={"color":"white"}
        )
        st.plotly_chart(gauge, use_container_width=True)

        # ── Summary table ──────────────────────────────────────
        st.markdown("### 📋 Full Results Table")
        tbl = pd.DataFrame([{
            "Model":            mname,
            "Verdict":          "✅ Pass" if r["pred"]==1 else "⚠️ At Risk",
            "Pass Probability": f"{r['pass_pct']}%",
            "Risk Score":       f"{r['risk_pct']}%",
            "Selected":         "⭐" if mname==sel_name else "",
        } for mname,r in results.items()]).set_index("Model")
        st.dataframe(tbl, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "compare":
    banner("Model Comparison",
           "Performance metrics and ROC curves for all 4 models", "⚖️")

    with st.spinner("Training models..."):
        trained, feature_cols, category_maps, feature_means = \
            train_all_models(len(master))

    if trained is None:
        st.error("Training failed.")
        st.stop()

    from sklearn.metrics import roc_curve, confusion_matrix
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Accuracy / AUC table ──────────────────────────────────
    st.markdown("### 📊 Model Performance Summary")
    comp = pd.DataFrame({
        n: {"Accuracy":f"{v['acc']}%","ROC-AUC":f"{v['auc']}"}
        for n,v in trained.items()
    }).T
    st.dataframe(comp, use_container_width=True)

    # ── ROC curves ────────────────────────────────────────────
    st.markdown("### 📈 ROC Curves — All 4 Models")
    roc_clrs = ["#4f8ef7","#2ecc71","#ffb347","#a78bfa"]
    fig_roc  = go.Figure()
    for i,(n,v) in enumerate(trained.items()):
        fpr,tpr,_ = roc_curve(v["y_test"], v["y_prob"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{n} (AUC={v['auc']})",
            line=dict(color=roc_clrs[i%4], width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1],y=[0,1],name="Random",
        line=dict(color="grey",dash="dash")
    ))
    fig_roc.update_layout(
        template=TPL, xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Model info cards ──────────────────────────────────────
    st.markdown("### 🧠 How Each Model Works")
    model_info = {
        "🔵 Logistic Regression": ("#4f8ef7","Linear separator",
            "Draws a decision boundary as a line/plane through the data. Uses a sigmoid function to output probabilities. Fast, interpretable — good baseline."),
        "🌲 Random Forest":        ("#2ecc71","150 decision trees voting",
            "Builds 150 trees in parallel, each on random data subsets. Final prediction = majority vote. Robust, gives feature importance rankings."),
        "📈 Gradient Boosting":    ("#ffb347","Sequential error correction",
            "Builds trees one by one. Each tree corrects the mistakes of the previous. High accuracy on structured data. Slower to train."),
        "⚡ Hist Gradient Boost":  ("#a78bfa","Fast histogram boosting",
            "sklearn's best booster. Uses histograms to speed up splits. Handles NaN natively. Comparable to XGBoost in accuracy with zero compatibility issues."),
    }
    mc = st.columns(4)
    for i,(n,v) in enumerate(trained.items()):
        clr, algo, desc = model_info.get(n,("#4f8ef7","ML",""))
        mc[i].markdown(f"""
<div class="mcard" style="border-left:3px solid {clr}">
  <p class="mn" style="color:{clr}">{n}</p>
  <p style="color:#4f8ef7;font-size:11px;font-weight:600;margin:0 0 6px">{algo}</p>
  <p class="md">{desc}</p>
  <div style="display:flex;gap:12px;margin-top:12px">
    <div style="text-align:center">
      <p style="color:#6b7a99;font-size:10px;margin:0">ACCURACY</p>
      <p class="ms" style="color:{clr}">{v['acc']}%</p>
    </div>
    <div style="text-align:center">
      <p style="color:#6b7a99;font-size:10px;margin:0">ROC-AUC</p>
      <p class="ms" style="color:{clr}">{v['auc']}</p>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "data":
    banner("Data Explorer",
           f"Raw master dataset — {len(master):,} students × {master.shape[1]} columns",
           "🔍")

    col_filter = st.multiselect(
        "Select columns to display:",
        master.columns.tolist(),
        default=["gender","age_band","highest_education",
                 "final_result","total_clicks",
                 "avg_assessment_score","active_days"]
    )
    n_rows = st.slider("Rows to show", 10, 500, 50, 10)
    result_filter = st.multiselect(
        "Filter by result:",
        master["final_result"].unique(),
        default=list(master["final_result"].unique())
    )
    view = master[master["final_result"].isin(result_filter)]
    if col_filter:
        view = view[col_filter]
    st.dataframe(view.head(n_rows), use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Rows",    f"{len(master):,}")
    c2.metric("Total Columns", master.shape[1])
    c3.metric("Missing Values",int(master.isnull().sum().sum()))

    st.markdown("#### 📊 Column Statistics")
    st.dataframe(master.describe().round(2), use_container_width=True)


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#2d3a4a;font-size:12px'>"
    "🎓 OULAD Analytics Dashboard &nbsp;|&nbsp; "
    "Python + Streamlit &nbsp;|&nbsp; Open University Dataset"
    "</p>",
    unsafe_allow_html=True
)
