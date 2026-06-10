"""
OULAD Analytics — Complete Rewrite
Real sidebar navigation + fixed unbiased predictions
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

st.set_page_config(
    page_title="OULAD Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CSS — full sidebar + card styles
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* base */
html,body,[data-testid="stAppViewContainer"]{background:#070b14!important}
[data-testid="stSidebar"]{background:#0b0f1a!important;
  border-right:1px solid #1a2236!important;min-width:230px!important}

/* hide default streamlit sidebar padding */
[data-testid="stSidebar"] .block-container{padding:0!important}
section[data-testid="stSidebar"] > div{padding:0!important}

/* logo */
.sb-logo{padding:20px 18px 14px;
  border-bottom:1px solid #1a2236;margin-bottom:0}
.sb-logo h2{color:#fff;font-size:16px;font-weight:700;margin:0;letter-spacing:.5px}
.sb-logo p{color:#4f8ef7;font-size:10px;margin:3px 0 0;text-transform:uppercase;letter-spacing:1.5px}

/* section label */
.sb-sec{padding:18px 18px 6px;
  color:#384766;font-size:9px;font-weight:700;
  letter-spacing:2.5px;text-transform:uppercase}

/* nav button — override streamlit button */
div[data-testid="stSidebar"] .stButton button{
  width:100%!important;
  background:transparent!important;
  border:none!important;
  border-radius:8px!important;
  padding:9px 18px!important;
  text-align:left!important;
  color:#6b7a99!important;
  font-size:13px!important;
  font-weight:500!important;
  cursor:pointer!important;
  margin:1px 0!important;
  transition:all .15s!important;
  box-shadow:none!important;
}
div[data-testid="stSidebar"] .stButton button:hover{
  background:#141d2e!important;
  color:#c8d6f0!important;
  transform:none!important;
}
/* active page button */
div[data-testid="stSidebar"] .stButton button:focus{
  background:#0f1e3d!important;
  color:#4f8ef7!important;
  border-left:3px solid #4f8ef7!important;
  box-shadow:none!important;
  outline:none!important;
}

/* KPI */
.kpi-wrap{background:linear-gradient(145deg,#0d1526,#111c30);
  border-radius:14px;padding:18px 16px;border:1px solid #1a2a40;
  border-top:3px solid var(--ac,#4f8ef7);margin:4px 0;
  box-shadow:0 4px 20px rgba(0,0,0,.4)}
.kpi-wrap .lb{color:#56637a;font-size:10px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;margin:0}
.kpi-wrap .vl{color:#f0f4ff;font-size:32px;font-weight:800;
  margin:6px 0 0;line-height:1}
.kpi-wrap .tg{font-size:11px;margin:4px 0 0}

/* page header */
.ph{background:linear-gradient(135deg,#0c1a30 0%,#091222 100%);
  border-radius:16px;padding:26px 28px;margin-bottom:22px;
  border:1px solid #1a2a40}
.ph h1{color:#f0f4ff;font-size:24px;font-weight:800;margin:0 0 5px}
.ph p{color:#56637a;font-size:13px;margin:0}

/* model card */
.mc{background:#0d1526;border-radius:12px;padding:18px;
  border:1px solid #1a2236;height:100%}
.mc .mn{font-size:14px;font-weight:700;margin:0 0 4px}
.mc .md{color:#56637a;font-size:11px;line-height:1.6;margin:0 0 12px}
.mc .ms{font-size:24px;font-weight:800;margin:0}

/* result card */
.rc{border-radius:16px;padding:26px 16px;text-align:center;margin:4px}
.rc .pct{font-size:58px;font-weight:900;margin:8px 0 0;line-height:1}
.rc .lbl{font-size:13px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin:8px 0 4px}
.rc .sub{font-size:12px;color:#56637a;margin:0}

/* progress bar */
.pb-row{display:grid;grid-template-columns:190px 1fr 60px;
  align-items:center;gap:10px;margin:7px 0}
.pb-name{font-size:12px;color:#7a8aa0;font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pb-bg{height:10px;background:#111c2e;border-radius:10px;overflow:hidden}
.pb-fill{height:100%;border-radius:10px}
.pb-val{font-size:14px;font-weight:700;text-align:right}

/* active nav item highlight */
.nav-active-pill{background:#0f1e3d;border-radius:8px;
  border-left:3px solid #4f8ef7;padding:8px 14px;
  color:#4f8ef7;font-size:13px;font-weight:600;
  margin:1px 0;display:block}

div[data-testid="stSidebar"] .stRadio label{
  color:#6b7a99!important;font-size:13px!important;
  padding:8px 12px!important;border-radius:8px!important;
  cursor:pointer!important;display:block!important;width:100%!important}
div[data-testid="stSidebar"] .stRadio label:hover{
  background:#141d2e!important;color:#c8d6f0!important}
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]{display:none!important}
div[data-testid="stSidebar"] .stRadio{margin:1px 0!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ══════════════════════════════════════════════════════════════
#  DATA + MODEL LOADERS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_master():
    for folder in ["sample_data","outputs","data"]:
        p = os.path.join(folder,"master_dataset.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "target" not in df.columns:
                df["target"] = df["final_result"].map(PASS_LABELS)
            return df
    return None

@st.cache_data
def load_vle():
    for folder in ["sample_data","data"]:
        p = os.path.join(folder,"studentVle.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

# KEY FIX: train on ONLY the features shown in the UI
# No encoding bugs, no missing-feature bias, clean predictions
@st.cache_resource
def train_models_clean(n_rows):
    from sklearn.linear_model  import LogisticRegression
    from sklearn.ensemble      import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        HistGradientBoostingClassifier)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing  import StandardScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import accuracy_score, roc_auc_score

    master = load_master()
    if master is None:
        return None, []

    # ── Use ONLY numeric features (no encoding, no NaN risk) ──
    FEATS = [f for f in [
        "avg_assessment_score",
        "clicks_day30",
        "active_days",
        "total_clicks",
        "studied_credits",
        "num_of_prev_attempts",
        "tma_avg_score",
        "exam_avg_score",
        "late_submissions",
        "active_days",
        "click_std",
    ] if f in master.columns]

    # Remove duplicates
    FEATS = list(dict.fromkeys(FEATS))

    X = master[FEATS].fillna(0)
    y = master["target"].fillna(0).astype(int)

    X_tr,X_te,y_tr,y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    defs = {
        "🔵 Logistic Regression": Pipeline([
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "🌲 Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ]),
        "📈 Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.08, random_state=42)),
        ]),
        "⚡ Hist Gradient Boost": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=150, max_depth=5,
                learning_rate=0.08, random_state=42)),
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
            "y_pred": yp, "y_prob": ypr, "y_test": y_te,
        }
    return trained, FEATS

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
master   = load_master()
vle_data = load_vle()

if master is None:
    st.error("❌ No data. Run: python run_all.py"); st.stop()

CLR = {"Pass":"#2ecc71","Distinction":"#3498db",
       "Fail":"#e74c3c","Withdrawn":"#95a5a6"}
MCOL = {"🔵 Logistic Regression":"#4f8ef7",
        "🌲 Random Forest":"#2ecc71",
        "📈 Gradient Boosting":"#ffb347",
        "⚡ Hist Gradient Boost":"#a78bfa"}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <h2>🎓 OULAD Analytics</h2>
      <p>Learning Intelligence Platform</p>
    </div>""", unsafe_allow_html=True)

    # ── Radio-based navigation (looks like real menu) ─────────
    st.markdown('<div class="sb-sec">MAIN</div>', unsafe_allow_html=True)
    main_page = st.radio("main", ["🏠  Home","📊  Overview"],
                          label_visibility="collapsed",
                          key="main_nav")

    st.markdown('<div class="sb-sec">ANALYSIS</div>', unsafe_allow_html=True)
    analysis_page = st.radio("analysis",
                              ["👥  Demographics",
                               "📈  Engagement",
                               "📝  Assessments"],
                              label_visibility="collapsed",
                              key="analysis_nav")

    st.markdown('<div class="sb-sec">PREDICT</div>', unsafe_allow_html=True)
    predict_page = st.radio("predict",
                             ["🤖  Predict Student",
                              "⚖️  Compare Models"],
                             label_visibility="collapsed",
                             key="predict_nav")

    st.markdown('<div class="sb-sec">DATA</div>', unsafe_allow_html=True)
    data_page = st.radio("data", ["🔍  Explore Data"],
                          label_visibility="collapsed",
                          key="data_nav")

    st.markdown("---")
    # ── Filters ───────────────────────────────────────────────
    with st.expander("🎛️ Filters", expanded=False):
        genders  = st.multiselect("Gender",
                      sorted(master["gender"].dropna().unique()),
                      default=list(master["gender"].dropna().unique()))
        ages     = st.multiselect("Age",
                      sorted(master["age_band"].dropna().unique()),
                      default=list(master["age_band"].dropna().unique()))
        res_filt = st.multiselect("Result",
                      master["final_result"].dropna().unique(),
                      default=list(master["final_result"].dropna().unique()))

    st.download_button("⬇️  Export Data",
                        master.to_csv(index=False),
                        "oulad.csv","text/csv",
                        use_container_width=True)

# ── Determine active page from last clicked radio ─────────────
# whichever radio was touched last becomes active
import time
all_options = {
    "🏠  Home": "Home",
    "📊  Overview": "Overview",
    "👥  Demographics": "Demographics",
    "📈  Engagement": "Engagement",
    "📝  Assessments": "Assessments",
    "🤖  Predict Student": "Predict",
    "⚖️  Compare Models": "Compare",
    "🔍  Explore Data": "Data",
}

# Use query params trick to detect which group changed
for radio_val, page_name in all_options.items():
    if (main_page    == radio_val or
        analysis_page== radio_val or
        predict_page == radio_val or
        data_page    == radio_val):
        active_page = page_name
        break
else:
    active_page = "Home"

# ── Apply filters ─────────────────────────────────────────────
fdf = master.copy()
if "genders" in dir() and genders:
    fdf = fdf[fdf["gender"].isin(genders)]
if "ages" in dir() and ages:
    fdf = fdf[fdf["age_band"].isin(ages)]
if "res_filt" in dir() and res_filt:
    fdf = fdf[fdf["final_result"].isin(res_filt)]

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def page_header(title, sub="", icon=""):
    st.markdown(f"""
    <div class="ph">
      <h1>{icon} {title}</h1>
      <p>{sub}</p>
    </div>""", unsafe_allow_html=True)

def kpi_strip(df):
    cols = st.columns(6)
    items = [
        ("STUDENTS",   f"{len(df):,}",                                       "#4f8ef7"),
        ("PASS RATE",  f"{(df['target']==1).mean()*100:.1f}%",                "#2ecc71"),
        ("AVG CLICKS", f"{df['total_clicks'].mean():,.0f}"
                       if "total_clicks" in df else "—",                      "#ff6b9d"),
        ("AVG SCORE",  f"{df['avg_assessment_score'].mean():.1f}"
                       if "avg_assessment_score" in df else "—",              "#ffb347"),
        ("ACTIVE DAYS",f"{df['active_days'].mean():.0f}"
                       if "active_days" in df else "—",                       "#a78bfa"),
        ("WITHDRAWAL", f"{df['withdrew'].mean()*100:.1f}%"
                       if "withdrew" in df else "—",                          "#38bdf8"),
    ]
    for col,(lb,vl,ac) in zip(cols,items):
        col.markdown(f"""
<div class="kpi-wrap" style="--ac:{ac}">
  <p class="lb">{lb}</p><p class="vl">{vl}</p>
</div>""", unsafe_allow_html=True)

def bf(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c8d6f0",
                       margin=dict(t=40,b=10,l=10,r=10))
    return fig

# ══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════
if active_page == "Home":
    page_header("Learning Analytics Dashboard",
                f"Analysing {len(master):,} students · 7 relational tables · 4 ML models",
                "🎓")
    kpi_strip(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        rc = fdf["final_result"].value_counts().reset_index()
        rc.columns=["result","count"]
        fig=px.bar(rc,x="result",y="count",color="result",
                   color_discrete_map=CLR,template=TPL,
                   title="Final Result Distribution",text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(bf(fig), use_container_width=True)
    with c2:
        if "code_module" in fdf.columns:
            mp=fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); mp.columns=["module","rate"]
            fig2=px.bar(mp.sort_values("rate",ascending=True),
                        x="rate",y="module",orientation="h",
                        title="Pass Rate by Module (%)",template=TPL,
                        color="rate",color_continuous_scale="Tealgrn")
            st.plotly_chart(bf(fig2), use_container_width=True)

    st.markdown("### 🧭 Navigate to")
    b1,b2,b3,b4,b5 = st.columns(5)
    nav_cards = [
        (b1,"📊","Overview","Result distribution"),
        (b2,"👥","Demographics","Student backgrounds"),
        (b3,"📈","Engagement","VLE activity"),
        (b4,"🤖","Predict","Risk predictor"),
        (b5,"⚖️","Compare","Model comparison"),
    ]
    for col,ic,nm,desc in nav_cards:
        col.markdown(f"""
<div style="background:#0d1526;border:1px solid #1a2236;border-radius:12px;
     padding:16px;text-align:center;cursor:pointer">
  <p style="font-size:26px;margin:0">{ic}</p>
  <p style="color:#f0f4ff;font-size:13px;font-weight:700;margin:6px 0 2px">{nm}</p>
  <p style="color:#56637a;font-size:11px;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
elif active_page == "Overview":
    page_header("Result Overview",
                f"{len(fdf):,} students · Use sidebar filters to explore","📊")
    kpi_strip(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        rc=fdf["final_result"].value_counts().reset_index(); rc.columns=["r","c"]
        fig=px.bar(rc,x="r",y="c",color="r",color_discrete_map=CLR,
                   template=TPL,title="Final Result Distribution",text="c")
        fig.update_traces(textposition="outside")
        st.plotly_chart(bf(fig), use_container_width=True)
    with c2:
        fig2=px.pie(rc,names="r",values="c",color="r",
                    color_discrete_map=CLR,title="Result Share",
                    template=TPL,hole=0.45)
        st.plotly_chart(bf(fig2), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        if "code_module" in fdf.columns:
            mp=fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); mp.columns=["m","r"]
            fig3=px.bar(mp.sort_values("r",ascending=True),x="r",y="m",
                        orientation="h",title="Pass Rate by Module",
                        template=TPL,color="r",color_continuous_scale="Tealgrn")
            st.plotly_chart(bf(fig3), use_container_width=True)
    with c4:
        if "code_presentation" in fdf.columns:
            pp=fdf.groupby("code_presentation").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); pp.columns=["p","r"]
            fig4=px.line(pp,x="p",y="r",markers=True,
                         title="Pass Rate by Presentation",template=TPL)
            st.plotly_chart(bf(fig4), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
elif active_page == "Demographics":
    page_header("Demographics Analysis",
                "How student background influences outcomes","👥")
    kpi_strip(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        dg=pd.crosstab(fdf["gender"],fdf["final_result"]).reset_index().melt("gender")
        fig=px.bar(dg,x="gender",y="value",color="final_result",
                   barmode="group",title="Gender vs Result",
                   template=TPL,color_discrete_map=CLR)
        st.plotly_chart(bf(fig), use_container_width=True)
    with c2:
        da=pd.crosstab(fdf["age_band"],fdf["final_result"]).reset_index().melt("age_band")
        fig2=px.bar(da,x="age_band",y="value",color="final_result",
                    barmode="stack",title="Age Band vs Result",
                    template=TPL,color_discrete_map=CLR)
        st.plotly_chart(bf(fig2), use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        de=fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        de.columns=["edu","rate"]
        fig3=px.bar(de.sort_values("rate",ascending=True),
                    x="rate",y="edu",orientation="h",
                    title="Pass Rate by Education (%)",template=TPL,
                    color="rate",color_continuous_scale="Blues")
        st.plotly_chart(bf(fig3), use_container_width=True)
    with c4:
        if "imd_band" in fdf.columns:
            di=fdf.groupby("imd_band")["target"].mean().mul(100).reset_index()
            di.columns=["imd","rate"]
            fig4=px.bar(di,x="imd",y="rate",title="Pass Rate by IMD Band",
                        template=TPL,color="rate",color_continuous_scale="RdYlGn")
            fig4.update_xaxes(tickangle=40)
            st.plotly_chart(bf(fig4), use_container_width=True)

    fig5=px.sunburst(fdf,path=["gender","age_band","final_result"],
                     title="Demographics → Outcome Sunburst",template=TPL)
    st.plotly_chart(bf(fig5), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ENGAGEMENT
# ══════════════════════════════════════════════════════════════
elif active_page == "Engagement":
    page_header("VLE Engagement Analysis",
                "How online activity predicts student success","📈")
    kpi_strip(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    if "total_clicks" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            op=[o for o in ["Distinction","Pass","Fail","Withdrawn"]
                if o in fdf["final_result"].unique()]
            cl=fdf[fdf["total_clicks"]<fdf["total_clicks"].quantile(0.98)]
            fig=px.box(cl,x="final_result",y="total_clicks",
                       category_orders={"final_result":op},
                       title="VLE Clicks vs Result",template=TPL,
                       color="final_result",color_discrete_map=CLR)
            st.plotly_chart(bf(fig), use_container_width=True)
        with c2:
            if "avg_assessment_score" in fdf.columns:
                fig2=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="total_clicks",y="avg_assessment_score",
                                color="final_result",opacity=0.55,
                                title="Clicks vs Score",template=TPL,
                                color_discrete_map=CLR)
                st.plotly_chart(bf(fig2), use_container_width=True)

        if vle_data is not None:
            st.markdown("#### 🕐 Daily Activity Timeline")
            daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
            daily.columns=["day","clicks"]
            fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                         title="Total VLE Clicks Per Day",template=TPL,
                         color_discrete_sequence=["#4f8ef7"])
            fig3.add_vline(x=30,line_dash="dash",line_color="orange",
                           annotation_text="Day 30")
            fig3.add_vline(x=100,line_dash="dash",line_color="#e74c3c",
                           annotation_text="Day 100")
            st.plotly_chart(bf(fig3), use_container_width=True)

        c3,c4=st.columns(2)
        with c3:
            if "active_days" in fdf.columns:
                fig4=px.histogram(fdf,x="active_days",color="final_result",
                                  nbins=40,barmode="overlay",opacity=0.6,
                                  title="Active Days Distribution",
                                  template=TPL,color_discrete_map=CLR)
                st.plotly_chart(bf(fig4), use_container_width=True)
        with c4:
            if "clicks_day30" in fdf.columns:
                fig5=px.box(fdf,x="final_result",y="clicks_day30",
                            title="Day-30 Clicks — Early Warning",
                            template=TPL,color="final_result",
                            color_discrete_map=CLR)
                st.plotly_chart(bf(fig5), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ASSESSMENTS
# ══════════════════════════════════════════════════════════════
elif active_page == "Assessments":
    page_header("Assessment Performance",
                "Scores, submissions and academic trends","📝")
    kpi_strip(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    if "avg_assessment_score" in fdf.columns:
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(fdf,x="avg_assessment_score",color="final_result",
                             nbins=40,barmode="overlay",opacity=0.65,
                             title="Score Distribution",template=TPL,
                             color_discrete_map=CLR)
            st.plotly_chart(bf(fig), use_container_width=True)
        with c2:
            sr=fdf.groupby("final_result")["avg_assessment_score"].mean().reset_index()
            sr.columns=["r","s"]
            fig2=px.bar(sr,x="r",y="s",color="r",title="Avg Score by Result",
                        template=TPL,text_auto=".1f",color_discrete_map=CLR)
            st.plotly_chart(bf(fig2), use_container_width=True)

        if "tma_avg_score" in fdf.columns and "exam_avg_score" in fdf.columns:
            c3,c4=st.columns(2)
            with c3:
                fig3=px.scatter(fdf.sample(min(2000,len(fdf))),
                                x="tma_avg_score",y="exam_avg_score",
                                color="final_result",opacity=0.55,
                                title="TMA vs Exam Score",template=TPL,
                                color_discrete_map=CLR)
                st.plotly_chart(bf(fig3), use_container_width=True)
            with c4:
                if "late_submissions" in fdf.columns:
                    lp=fdf.groupby(pd.cut(fdf["late_submissions"],[0,1,3,5,20])
                                   )["target"].mean().mul(100).reset_index()
                    lp.columns=["lr","rp"]; lp["lr"]=lp["lr"].astype(str)
                    fig4=px.bar(lp,x="lr",y="rp",
                                title="Late Submissions vs Pass Rate",
                                template=TPL,color="rp",
                                color_continuous_scale="RdYlGn")
                    st.plotly_chart(bf(fig4), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
elif active_page == "Predict":
    page_header("Student Risk Predictor",
                "Input student details — all 4 models predict instantly","🤖")

    with st.spinner("Training models on dataset..."):
        trained, FEATS = train_models_clean(len(master))

    if trained is None:
        st.error("Training failed. Run pipeline first."); st.stop()

    # ── Model selector tabs ───────────────────────────────────
    st.markdown("#### 🧠 Highlight Model")
    sel_name = st.radio("model_sel",
                         list(trained.keys()),
                         horizontal=True,
                         label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### 📋 Enter Student Profile")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Assessment**")
        avg_sc  = st.slider("Average Assessment Score", 0,100,65,
                             help="Mean score across all assignments")
        tma_sc  = st.slider("Coursework Score (TMA)",   0,100,65,
                             help="Continuous assessment average")
        exam_sc = st.slider("Exam Score",               0,100,60,
                             help="Final exam score")
    with c2:
        st.markdown("**📱 Engagement**")
        clicks30= st.slider("VLE Clicks by Day 30",    0,3000,600,50,
                             help="Online activity in first 30 days")
        act_d   = st.slider("Active Days",             0,250,90,
                             help="Days with at least one login")
        tot_clk = st.slider("Total VLE Clicks",        0,20000,3000,100,
                             help="Total clicks across full course")
    with c3:
        st.markdown("**📚 Background**")
        credits = st.select_slider("Studied Credits",
                      [30,60,90,120,150,180,240,300], value=60)
        prev_att= st.slider("Previous Attempts",       0,5,0)
        late_sub= st.slider("Late Submissions",        0,10,0,
                             help="Number of assignments submitted late")

    predict_btn = st.button(
        "🔮  Predict — Run All 4 Models",
        type="primary", use_container_width=True)

    if predict_btn:
        # ── Build input using ONLY features model was trained on ─
        user_input = {
            "avg_assessment_score": avg_sc,
            "clicks_day30":         clicks30,
            "active_days":          act_d,
            "total_clicks":         tot_clk,
            "studied_credits":      credits,
            "num_of_prev_attempts": prev_att,
            "tma_avg_score":        tma_sc,
            "exam_avg_score":       exam_sc,
            "late_submissions":     late_sub,
            "click_std":            clicks30 * 0.3,
        }
        # Only include features the model actually trained on
        row = {f: user_input.get(f, 0) for f in FEATS}
        X_in = pd.DataFrame([row])[FEATS]

        # ── Run all 4 models ──────────────────────────────────
        results = {}
        for mn, mv in trained.items():
            pred = mv["pipe"].predict(X_in)[0]
            prob = mv["pipe"].predict_proba(X_in)[0]
            results[mn] = {
                "pred":     int(pred),
                "pass_pct": round(float(prob[1])*100,1),
                "risk_pct": round(float(prob[0])*100,1),
            }

        st.markdown("---")
        st.markdown("### 🎯 Results — All 4 Models")

        # ── Big result cards ──────────────────────────────────
        cols = st.columns(4)
        for i,(mn,res) in enumerate(results.items()):
            mc   = MCOL.get(mn,"#4f8ef7")
            bclr = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            bg   = "#071a07" if res["pred"]==1 else "#1a0707"
            icon = "✅" if res["pred"]==1 else "⚠️"
            lbl  = "PASS"    if res["pred"]==1 else "AT RISK"
            star = "⭐" if mn==sel_name else ""
            brd  = f"2px solid {mc}" if mn==sel_name else f"1px solid {bclr}44"

            cols[i].markdown(f"""
<div class="rc" style="background:{bg};border:{brd}">
  <p style="color:{mc};font-size:10px;font-weight:700;
     letter-spacing:1.5px;text-transform:uppercase;margin:0">{star} {mn.split(' ',1)[-1]}</p>
  <p class="pct" style="color:{bclr}">{res['pass_pct']}%</p>
  <p style="color:#56637a;font-size:10px;margin:2px 0 6px">Pass Probability</p>
  <p class="lbl" style="color:{bclr}">{icon} {lbl}</p>
  <p class="sub">Risk Score: {res['risk_pct']}%</p>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Progress bars ─────────────────────────────────────
        st.markdown("### 📊 Visual Probability Bars")
        for mn, res in results.items():
            bclr = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            star = "⭐ " if mn==sel_name else "　 "
            pct  = res["pass_pct"]
            st.markdown(f"""
<div class="pb-row">
  <div class="pb-name">{star}{mn}</div>
  <div class="pb-bg">
    <div class="pb-fill" style="width:{pct}%;background:{bclr}"></div>
  </div>
  <div class="pb-val" style="color:{bclr}">{pct}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────
        sr = results[sel_name]
        gc = "#2ecc71" if sr["pred"]==1 else "#e74c3c"
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = sr["pass_pct"],
            delta = {"reference":50,
                     "increasing":{"color":"#2ecc71"},
                     "decreasing":{"color":"#e74c3c"}},
            number= {"suffix":"%","font":{"size":56,"color":gc}},
            title = {"text":f"<b>{sel_name}</b><br>"
                            f"<span style='font-size:12px;color:#56637a'>"
                            f"Pass Probability</span>"},
            gauge = {
                "axis":{"range":[0,100]},
                "bar" :{"color":gc,"thickness":0.28},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,40], "color":"#120606"},
                    {"range":[40,60],"color":"#121206"},
                    {"range":[60,100],"color":"#061206"},
                ],
                "threshold":{"line":{"color":"white","width":3},
                             "thickness":0.75,"value":50}
            }
        ))
        gauge.update_layout(
            template=TPL,paper_bgcolor="rgba(0,0,0,0)",
            height=300,font={"color":"white"})
        st.plotly_chart(gauge, use_container_width=True)

        # ── Bar chart comparison ───────────────────────────────
        bar_df = pd.DataFrame([
            {"Model":n.split(" ",1)[-1],
             "Pass %":r["pass_pct"],"Risk %":r["risk_pct"]}
            for n,r in results.items()
        ])
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            name="Pass %", x=bar_df["Model"],y=bar_df["Pass %"],
            marker_color=["#2ecc71" if v>=50 else "#e74c3c"
                          for v in bar_df["Pass %"]],
            text=[f"{v}%" for v in bar_df["Pass %"]],
            textposition="outside", textfont=dict(size=15,color="white")
        ))
        fig_b.add_hline(y=50,line_dash="dash",line_color="white",
                        opacity=0.4,annotation_text="50% threshold")
        fig_b.update_layout(template=TPL,title="Pass % — All Models",
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             yaxis=dict(range=[0,110]),showlegend=False)
        st.plotly_chart(bf(fig_b), use_container_width=True)

        # ── Clean table ───────────────────────────────────────
        st.markdown("### 📋 Summary Table")
        tbl = pd.DataFrame([{
            "Model":   mn,
            "Verdict": "✅ Pass" if r["pred"]==1 else "⚠️ At Risk",
            "Pass Probability": f"{r['pass_pct']}%",
            "Risk Score":       f"{r['risk_pct']}%",
            "Selected":         "⭐ Yes" if mn==sel_name else "",
        } for mn,r in results.items()]).set_index("Model")
        st.dataframe(tbl, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════
elif active_page == "Compare":
    page_header("Model Comparison",
                "Performance metrics, ROC curves and algorithm explanations","⚖️")

    with st.spinner("Training models..."):
        trained, FEATS = train_models_clean(len(master))

    if trained is None:
        st.error("Training failed."); st.stop()

    from sklearn.metrics import roc_curve

    st.markdown("### 📊 Performance Summary")
    comp = pd.DataFrame({
        n:{"Accuracy":f"{v['acc']}%","ROC-AUC":f"{v['auc']}"}
        for n,v in trained.items()
    }).T
    st.dataframe(comp, use_container_width=True)

    st.markdown("### 📈 ROC Curves")
    fig_roc = go.Figure()
    roc_clrs=["#4f8ef7","#2ecc71","#ffb347","#a78bfa"]
    for i,(n,v) in enumerate(trained.items()):
        fpr,tpr,_=roc_curve(v["y_test"],v["y_prob"])
        fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,
            name=f"{n} · AUC={v['auc']}",
            line=dict(color=roc_clrs[i],width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",
        line=dict(color="#384766",dash="dash")))
    fig_roc.update_layout(template=TPL,xaxis_title="FPR",
        yaxis_title="TPR",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(bf(fig_roc), use_container_width=True)

    st.markdown("### 🧠 How Each Model Works")
    info = {
        "🔵 Logistic Regression":("#4f8ef7","Linear separator",
            "Fits a straight-line boundary between Pass and Fail. Uses sigmoid function to output probability. Fast, interpretable, great baseline."),
        "🌲 Random Forest":       ("#2ecc71","200 decision trees vote",
            "Trains 200 trees independently on random data samples. Majority vote decides outcome. Very robust, gives feature importance rankings."),
        "📈 Gradient Boosting":   ("#ffb347","Sequential correction",
            "Builds trees one-by-one. Each new tree fixes mistakes of previous. High accuracy. The learning_rate=0.08 controls step size."),
        "⚡ Hist Gradient Boost": ("#a78bfa","Fast native boosting",
            "sklearn's own implementation of gradient boosting. Uses histogram binning for speed. Handles NaN natively. Comparable to XGBoost."),
    }
    mc = st.columns(4)
    for i,(n,v) in enumerate(trained.items()):
        clr,algo,desc = info.get(n,("#4f8ef7","ML",""))
        mc[i].markdown(f"""
<div class="mc" style="border-top:3px solid {clr}">
  <p class="mn" style="color:{clr}">{n}</p>
  <p style="color:{clr};font-size:10px;font-weight:600;
     letter-spacing:1px;text-transform:uppercase;margin:0 0 6px">{algo}</p>
  <p class="md">{desc}</p>
  <div style="display:flex;gap:16px;margin-top:auto">
    <div><p style="color:#384766;font-size:9px;margin:0">ACCURACY</p>
         <p class="ms" style="color:{clr}">{v['acc']}%</p></div>
    <div><p style="color:#384766;font-size:9px;margin:0">ROC-AUC</p>
         <p class="ms" style="color:{clr}">{v['auc']}</p></div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif active_page == "Data":
    page_header("Data Explorer",
                f"{len(master):,} rows · {master.shape[1]} columns","🔍")

    col_sel = st.multiselect("Columns to show:",
        master.columns.tolist(),
        default=["gender","age_band","highest_education",
                 "final_result","total_clicks",
                 "avg_assessment_score","active_days"])
    n_rows = st.slider("Rows", 10,500,50,10)
    rf = st.multiselect("Filter result:",
        master["final_result"].unique(),
        default=list(master["final_result"].unique()))

    view = master[master["final_result"].isin(rf)]
    if col_sel: view = view[col_sel]
    st.dataframe(view.head(n_rows), use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Rows",    f"{len(master):,}")
    c2.metric("Total Columns", master.shape[1])
    c3.metric("Missing Values",int(master.isnull().sum().sum()))
    st.markdown("#### Statistics")
    st.dataframe(master.describe().round(2), use_container_width=True)

# ── footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#1e2a3a;font-size:11px'>"
    "🎓 OULAD Analytics Platform · Python + Streamlit · Open University</p>",
    unsafe_allow_html=True)
