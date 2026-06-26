"""
OULAD Analytics Dashboard - Clean Final Version
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

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#070b14!important}
[data-testid="stSidebar"]{background:#0b0e1a!important;
  border-right:1px solid #16213a!important;padding-top:0!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0!important}

/* hide all sidebar button borders/bg */
[data-testid="stSidebar"] .stButton button{
  background:transparent!important;border:none!important;
  box-shadow:none!important;padding:9px 16px!important;
  text-align:left!important;color:#6b7a99!important;
  font-size:13px!important;font-weight:500!important;
  border-radius:0!important;width:100%!important;transition:all .12s!important}
[data-testid="stSidebar"] .stButton button:hover{
  background:#141d30!important;color:#c8d6f0!important}
[data-testid="stSidebar"] .stButton button:focus{outline:none!important}

/* section label */
.snav{padding:16px 16px 4px;color:#253555;font-size:9px;font-weight:700;
  letter-spacing:3px;text-transform:uppercase}

/* active page pill */
.active-pill{background:#0f1e3d;border-left:3px solid #4f8ef7;
  padding:9px 16px;color:#4f8ef7!important;font-size:13px;
  font-weight:700;display:block;margin:1px 0}

/* KPI */
.kpi{background:linear-gradient(145deg,#0c1525,#111c30);
  border-radius:14px;padding:18px 14px;border:1px solid #16213a;
  border-top:3px solid var(--c,#4f8ef7);margin:3px 0}
.kpi .lb{color:#3a4d6b;font-size:10px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;margin:0}
.kpi .vl{color:#f0f4ff;font-size:30px;font-weight:900;margin:6px 0 0;line-height:1}

/* page header */
.ph{background:linear-gradient(135deg,#0c1830,#08111e);
  border-radius:16px;padding:26px 28px;margin-bottom:20px;
  border:1px solid #16213a}
.ph h1{color:#f0f4ff;font-size:24px;font-weight:800;margin:0 0 5px}
.ph p{color:#3a4d6b;font-size:13px;margin:0}

/* result card */
.rcard{border-radius:16px;padding:22px 12px;text-align:center;margin:3px}
.rcard .rpct{font-size:52px;font-weight:900;margin:8px 0 0;line-height:1}
.rcard .rlbl{font-size:11px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin:8px 0 4px}

/* progress bar */
.pbar{display:grid;grid-template-columns:185px 1fr 56px;
  align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #0c1525}
.pbar:last-child{border-bottom:none}
.pbar-name{font-size:12px;color:#6b7a99;font-weight:600}
.pbar-bg{height:11px;background:#0c1525;border-radius:11px;overflow:hidden}
.pbar-fill{height:100%;border-radius:11px}
.pbar-val{font-size:14px;font-weight:800;text-align:right}

/* model info card */
.mcard{background:#0c1525;border-radius:12px;padding:18px;
  border:1px solid #16213a;height:100%}

/* predict button */
div[data-testid="stMain"] .stButton button[kind="primary"]{
  background:linear-gradient(135deg,#1a56db,#4f8ef7)!important;
  color:#fff!important;font-size:18px!important;font-weight:700!important;
  padding:16px 0!important;border-radius:14px!important;border:none!important;
  box-shadow:0 4px 28px rgba(79,142,247,.4)!important;
  letter-spacing:.5px!important;transition:all .2s!important;
  width:100%!important}
div[data-testid="stMain"] .stButton button[kind="primary"]:hover{
  box-shadow:0 8px 36px rgba(79,142,247,.6)!important;
  transform:translateY(-2px)!important}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
cur = st.session_state.page

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
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

@st.cache_resource
def train_models(n_rows):
    from sklearn.linear_model    import LogisticRegression
    from sklearn.ensemble        import (RandomForestClassifier,
                                          GradientBoostingClassifier,
                                          HistGradientBoostingClassifier)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import accuracy_score, roc_auc_score

    df = load_master()
    if df is None: return None, []

    FEATS = [f for f in [
        "avg_assessment_score","clicks_day30","active_days","total_clicks",
        "studied_credits","num_of_prev_attempts","tma_avg_score",
        "exam_avg_score","late_submissions","click_std"
    ] if f in df.columns]

    X = df[FEATS].fillna(0)
    y = df["target"].fillna(0).astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,
                                        random_state=42,stratify=y)
    models = {
        "🔵 Logistic Regression": Pipeline([
            ("s",StandardScaler()),
            ("c",LogisticRegression(max_iter=1000,random_state=42))]),
        "🌲 Random Forest": Pipeline([
            ("c",RandomForestClassifier(n_estimators=200,max_depth=10,
                                         min_samples_leaf=5,n_jobs=-1,random_state=42))]),
        "📈 Gradient Boosting": Pipeline([
            ("c",GradientBoostingClassifier(n_estimators=150,max_depth=4,
                                              learning_rate=0.08,random_state=42))]),
        "⚡ Hist Gradient Boost": Pipeline([
            ("c",HistGradientBoostingClassifier(max_iter=150,max_depth=5,
                                                  learning_rate=0.08,random_state=42))]),
    }
    trained = {}
    for nm,pipe in models.items():
        pipe.fit(Xtr,ytr)
        yp  = pipe.predict(Xte)
        ypr = pipe.predict_proba(Xte)[:,1]
        trained[nm] = {"pipe":pipe,
                        "acc":round(accuracy_score(yte,yp)*100,1),
                        "auc":round(roc_auc_score(yte,ypr),4),
                        "y_pred":yp,"y_prob":ypr,"y_test":yte}
    return trained, FEATS

master   = load_master()
vle_data = load_vle()

if master is None:
    st.error("❌ No data found. Run: python run_all.py"); st.stop()

CLR  = {"Pass":"#2ecc71","Distinction":"#3498db",
        "Fail":"#e74c3c","Withdrawn":"#95a5a6"}
MCOL = {"🔵 Logistic Regression":"#4f8ef7","🌲 Random Forest":"#2ecc71",
        "📈 Gradient Boosting":"#ffb347","⚡ Hist Gradient Boost":"#a78bfa"}

# ─────────────────────────────────────────────────────────────
# SIDEBAR  — simple buttons, NO label_visibility
# ─────────────────────────────────────────────────────────────
NAV = {
    "MAIN":     [("🏠","Home"),("📊","Overview")],
    "ANALYSIS": [("👥","Demographics"),("📈","Engagement"),("📝","Assessments")],
    "PREDICT":  [("🤖","Predict"),("⚖️","Compare")],
    "DATA":     [("🔍","Explore Data")],
}

with st.sidebar:
    st.markdown("""
<div style="padding:20px 16px 14px;border-bottom:1px solid #16213a">
  <p style="color:#f0f4ff;font-size:17px;font-weight:800;margin:0">
    🎓 OULAD Analytics</p>
  <p style="color:#4f8ef7;font-size:10px;letter-spacing:2px;
     text-transform:uppercase;margin:4px 0 0">
     Learning Intelligence</p>
</div>""", unsafe_allow_html=True)

    for section, items in NAV.items():
        st.markdown(f'<div class="snav">{section}</div>',
                    unsafe_allow_html=True)
        for icon, name in items:
            if cur == name:
                # active page — show as styled pill
                st.markdown(f'<div class="active-pill">'
                             f'▶  {icon}  {name}</div>',
                             unsafe_allow_html=True)
            else:
                # inactive — plain button
                if st.button(f"    {icon}  {name}", key=f"nb_{name}"):
                    st.session_state.page = name
                    st.rerun()

    st.markdown("<hr style='border-color:#16213a;margin:14px 8px'>",
                unsafe_allow_html=True)

    with st.expander("🎛️  Filters"):
        genders = st.multiselect("Gender",
                      sorted(master["gender"].dropna().unique()),
                      default=list(master["gender"].dropna().unique()))
        ages    = st.multiselect("Age Band",
                      sorted(master["age_band"].dropna().unique()),
                      default=list(master["age_band"].dropna().unique()))
        res_f   = st.multiselect("Result",
                      master["final_result"].dropna().unique(),
                      default=list(master["final_result"].dropna().unique()))

    st.download_button("⬇️  Export CSV",
                        master.to_csv(index=False),
                        "oulad.csv","text/csv",use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
fdf = master.copy()
if genders: fdf = fdf[fdf["gender"].isin(genders)]
if ages:    fdf = fdf[fdf["age_band"].isin(ages)]
if res_f:   fdf = fdf[fdf["final_result"].isin(res_f)]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ph(title, sub="", icon=""):
    st.markdown(f"""<div class="ph">
    <h1>{icon} {title}</h1><p>{sub}</p></div>""",
    unsafe_allow_html=True)

def kpis(df):
    cols = st.columns(6)
    rows = [
        ("STUDENTS",   f"{len(df):,}",                                    "#4f8ef7"),
        ("PASS RATE",  f"{(df['target']==1).mean()*100:.1f}%",             "#2ecc71"),
        ("AVG CLICKS", f"{df['total_clicks'].mean():,.0f}"
                        if "total_clicks" in df.columns else "—",          "#ff6b9d"),
        ("AVG SCORE",  f"{df['avg_assessment_score'].mean():.1f}"
                        if "avg_assessment_score" in df.columns else "—",  "#ffb347"),
        ("ACTIVE DAYS",f"{df['active_days'].mean():.0f}"
                        if "active_days" in df.columns else "—",           "#a78bfa"),
        ("WITHDRAWAL", f"{df['withdrew'].mean()*100:.1f}%"
                        if "withdrew" in df.columns else "—",              "#38bdf8"),
    ]
    for col,(lb,vl,c) in zip(cols,rows):
        col.markdown(f"""<div class="kpi" style="--c:{c}">
        <p class="lb">{lb}</p><p class="vl">{vl}</p></div>""",
        unsafe_allow_html=True)

def bf(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c8d6f0",
                       margin=dict(t=40,b=10,l=10,r=10))
    return fig

cur = st.session_state.page

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
if cur == "Home":
    ph("Learning Analytics Dashboard",
       f"Open University · {len(master):,} students · 4 ML models","🎓")
    kpis(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        rc = fdf["final_result"].value_counts().reset_index()
        rc.columns = ["r","c"]
        fig = px.bar(rc,x="r",y="c",color="r",
                     color_discrete_map=CLR,template=TPL,
                     title="Final Result Distribution",text="c")
        fig.update_traces(textposition="outside")
        st.plotly_chart(bf(fig), use_container_width=True)
    with c2:
        if "code_module" in fdf.columns:
            mp = fdf.groupby("code_module").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); mp.columns=["m","r"]
            fig2 = px.bar(mp.sort_values("r",ascending=True),
                          x="r",y="m",orientation="h",
                          title="Pass Rate by Module (%)",template=TPL,
                          color="r",color_continuous_scale="Tealgrn")
            st.plotly_chart(bf(fig2), use_container_width=True)

    st.markdown("### 🧭 Quick Navigation")
    cols = st.columns(3)
    nav_cards = [
        ("📊","Overview",      "Result distribution",   "Overview",      "#4f8ef7"),
        ("👥","Demographics",  "Student backgrounds",   "Demographics",  "#2ecc71"),
        ("📈","Engagement",    "VLE activity",          "Engagement",    "#ffb347"),
        ("📝","Assessments",   "Score analysis",        "Assessments",   "#ff6b9d"),
        ("🤖","Predict",       "Risk predictor",        "Predict",       "#a78bfa"),
        ("⚖️","Compare",       "Model comparison",      "Compare",       "#38bdf8"),
    ]
    for i,(icon,name,desc,page,clr) in enumerate(nav_cards):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:#0c1525;border-radius:14px;padding:18px;
     border:1px solid #16213a;border-top:3px solid {clr};
     text-align:center;margin-bottom:4px">
  <p style="font-size:28px;margin:0">{icon}</p>
  <p style="color:#f0f4ff;font-size:14px;font-weight:700;margin:8px 0 3px">{name}</p>
  <p style="color:#3a4d6b;font-size:11px;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)
            if st.button(f"Open {name}", key=f"hc_{page}",
                         use_container_width=True):
                st.session_state.page = page
                st.rerun()

# ─────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────
elif cur == "Overview":
    ph("Result Overview", f"{len(fdf):,} students","📊")
    kpis(fdf)
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
                        orientation="h",title="Pass Rate by Module",template=TPL,
                        color="r",color_continuous_scale="Tealgrn")
            st.plotly_chart(bf(fig3), use_container_width=True)
    with c4:
        if "code_presentation" in fdf.columns:
            pp=fdf.groupby("code_presentation").apply(
                lambda x:(x["final_result"].isin(["Pass","Distinction"])).mean()*100
            ).reset_index(); pp.columns=["p","r"]
            fig4=px.line(pp,x="p",y="r",markers=True,
                         title="Pass Rate by Presentation",template=TPL)
            st.plotly_chart(bf(fig4), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────
elif cur == "Demographics":
    ph("Demographics Analysis","Student background vs outcomes","👥")
    kpis(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
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

    c3,c4 = st.columns(2)
    with c3:
        de=fdf.groupby("highest_education")["target"].mean().mul(100).reset_index()
        de.columns=["edu","rate"]
        fig3=px.bar(de.sort_values("rate",ascending=True),x="rate",y="edu",
                    orientation="h",title="Pass Rate by Education (%)",
                    template=TPL,color="rate",color_continuous_scale="Blues")
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
                     title="Demographics Sunburst",template=TPL)
    st.plotly_chart(bf(fig5), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ENGAGEMENT
# ─────────────────────────────────────────────────────────────
elif cur == "Engagement":
    ph("VLE Engagement","Online activity as a success predictor","📈")
    kpis(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    if "total_clicks" in fdf.columns:
        c1,c2 = st.columns(2)
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
            daily=vle_data.groupby("date")["sum_click"].sum().reset_index()
            daily.columns=["day","clicks"]
            fig3=px.area(daily.sort_values("day"),x="day",y="clicks",
                         title="Daily VLE Activity",template=TPL,
                         color_discrete_sequence=["#4f8ef7"])
            fig3.add_vline(x=30,line_dash="dash",line_color="orange",
                           annotation_text="Day 30")
            fig3.add_vline(x=100,line_dash="dash",line_color="#e74c3c",
                           annotation_text="Day 100")
            st.plotly_chart(bf(fig3), use_container_width=True)

        c3,c4 = st.columns(2)
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

# ─────────────────────────────────────────────────────────────
# ASSESSMENTS
# ─────────────────────────────────────────────────────────────
elif cur == "Assessments":
    ph("Assessment Performance","Scores and submission behaviour","📝")
    kpis(fdf)
    st.markdown("<br>", unsafe_allow_html=True)

    if "avg_assessment_score" in fdf.columns:
        c1,c2 = st.columns(2)
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
            c3,c4 = st.columns(2)
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

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
elif cur == "Predict":
    ph("Student Risk Predictor",
       "Enter student details — all 4 models predict instantly","🤖")

    with st.spinner("Loading models..."):
        trained, FEATS = train_models(len(master))
    if trained is None:
        st.error("Training failed. Run pipeline first."); st.stop()

    sel = st.radio("Highlight model:",list(trained.keys()),
                   index=2,horizontal=True,label_visibility="collapsed")

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Assessment Scores**")
        avg_sc  = st.slider("Average Score",    0,100,65,key="p1")
        tma_sc  = st.slider("Coursework (TMA)", 0,100,65,key="p2")
        exam_sc = st.slider("Exam Score",       0,100,60,key="p3")
    with c2:
        st.markdown("**📱 Online Engagement**")
        clicks30 = st.slider("Clicks by Day 30", 0,3000,600,50,key="p4")
        act_d    = st.slider("Active Days",       0,250,90,key="p5")
        tot_clk  = st.slider("Total VLE Clicks",  0,20000,3000,100,key="p6")
    with c3:
        st.markdown("**📚 Background**")
        credits  = st.select_slider("Credits",
                     [30,60,90,120,150,180,240,300],value=60,key="p7")
        prev_att = st.slider("Previous Attempts", 0,5,0,key="p8")
        late_sub = st.slider("Late Submissions",  0,10,0,key="p9")

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔮  Predict — All 4 Models",
                             type="primary",use_container_width=True)

    if predict_btn:
        inp = {"avg_assessment_score":avg_sc,"clicks_day30":clicks30,
               "active_days":act_d,"total_clicks":tot_clk,
               "studied_credits":credits,"num_of_prev_attempts":prev_att,
               "tma_avg_score":tma_sc,"exam_avg_score":exam_sc,
               "late_submissions":late_sub,"click_std":clicks30*0.3}
        row  = {f: inp.get(f,0) for f in FEATS}
        X_in = pd.DataFrame([row])[FEATS]

        results = {}
        for mn,mv in trained.items():
            try:
                pred = mv["pipe"].predict(X_in)[0]
                prob = mv["pipe"].predict_proba(X_in)[0]
                results[mn] = {"pred":int(pred),
                                "pass_pct":round(float(prob[1])*100,1),
                                "risk_pct":round(float(prob[0])*100,1)}
            except Exception:
                results[mn] = {"pred":0,"pass_pct":0.0,"risk_pct":100.0}

        st.markdown("---")
        st.markdown("### 🎯 All 4 Model Results")

        # result cards
        cols = st.columns(4)
        for i,(mn,res) in enumerate(results.items()):
            mc   = MCOL.get(mn,"#4f8ef7")
            bc   = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            bg   = "linear-gradient(145deg,#061606,#0b1e0b)" if res["pred"]==1 \
                   else "linear-gradient(145deg,#160606,#1e0b0b)"
            icon = "✅" if res["pred"]==1 else "⚠️"
            lbl  = "PASS" if res["pred"]==1 else "AT RISK"
            brd  = f"2px solid {mc}" if mn==sel else f"1px solid {bc}44"
            star = " ⭐" if mn==sel else ""
            cols[i].markdown(f"""
<div class="rcard" style="background:{bg};border:{brd}">
  <p style="color:{mc};font-size:10px;font-weight:700;
     letter-spacing:1.5px;text-transform:uppercase;margin:0">
     {mn.split(" ",1)[-1]}{star}</p>
  <p class="rpct" style="color:{bc}">{res["pass_pct"]}%</p>
  <p style="color:#3a4d6b;font-size:10px;margin:2px 0 8px">Pass Probability</p>
  <p class="rlbl" style="color:{bc}">{icon} {lbl}</p>
  <p style="color:#2d3d58;font-size:11px;margin:4px 0 0">
     Risk: {res["risk_pct"]}%</p>
</div>""", unsafe_allow_html=True)

        # progress bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Probability Bars")
        html = '<div style="background:#0c1525;border-radius:14px;padding:20px 22px;border:1px solid #16213a">'
        for mn,res in results.items():
            bc   = "#2ecc71" if res["pred"]==1 else "#e74c3c"
            star = "⭐ " if mn==sel else "　 "
            pct  = res["pass_pct"]
            html += f"""
<div class="pbar">
  <div class="pbar-name">{star}{mn}</div>
  <div class="pbar-bg">
    <div class="pbar-fill" style="width:{pct}%;background:{bc}"></div>
  </div>
  <div class="pbar-val" style="color:{bc}">{pct}%</div>
</div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        # summary table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Summary Table")
        tbl = pd.DataFrame([{
            "Model":   mn,
            "Verdict": "✅ Pass" if r["pred"]==1 else "⚠️ At Risk",
            "Pass %":  f"{r['pass_pct']}%",
            "Risk %":  f"{r['risk_pct']}%",
            "⭐":      "Yes" if mn==sel else "",
        } for mn,r in results.items()]).set_index("Model")
        st.dataframe(tbl, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# COMPARE
# ─────────────────────────────────────────────────────────────
elif cur == "Compare":
    ph("Model Comparison","Performance metrics and algorithm explanations","⚖️")

    with st.spinner("Training models..."):
        trained, FEATS = train_models(len(master))
    if trained is None:
        st.error("Training failed."); st.stop()

    st.markdown("### 📊 Performance Summary")
    comp = pd.DataFrame({
        n:{"Accuracy":f"{v['acc']}%","ROC-AUC":f"{v['auc']}"}
        for n,v in trained.items()}).T
    st.dataframe(comp, use_container_width=True)

    st.markdown("### 📈 ROC Curves")
    try:
        from sklearn.metrics import roc_curve
        fig_roc = go.Figure()
        for i,(n,v) in enumerate(trained.items()):
            fpr,tpr,_ = roc_curve(list(v["y_test"]),list(v["y_prob"]))
            fig_roc.add_trace(go.Scatter(
                x=list(fpr),y=list(tpr),
                name=f"{n.split(' ',1)[-1]} AUC={v['auc']}",
                line=dict(color=list(MCOL.values())[i],width=2.5)))
        fig_roc.add_trace(go.Scatter(
            x=[0,1],y=[0,1],name="Random",
            line=dict(color="#253555",dash="dash")))
        fig_roc.update_layout(
            template=TPL,height=400,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6f0")
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception as e:
        st.warning(f"ROC chart unavailable: {e}")

    st.markdown("### 🧠 Algorithm Explanations")
    info = {
        "🔵 Logistic Regression": ("#4f8ef7","Linear boundary",
            "Draws a straight decision boundary. Uses sigmoid to output probability 0-1. Fastest, fully interpretable."),
        "🌲 Random Forest":        ("#2ecc71","200 trees voting",
            "200 trees on random data subsets. Majority vote decides. Robust, gives feature importance."),
        "📈 Gradient Boosting":    ("#ffb347","Sequential correction",
            "Each tree corrects the previous one's errors. High accuracy on tabular data."),
        "⚡ Hist Gradient Boost":  ("#a78bfa","Fast sklearn boosting",
            "sklearn's native fast booster. Same accuracy as XGBoost, zero compatibility issues."),
    }
    mcols = st.columns(4)
    for i,(n,v) in enumerate(trained.items()):
        clr,algo,desc = info.get(n,("#4f8ef7","",""))
        mcols[i].markdown(f"""
<div class="mcard" style="border-top:3px solid {clr}">
  <p style="color:{clr};font-size:13px;font-weight:700;margin:0 0 3px">{n}</p>
  <p style="color:{clr};font-size:10px;font-weight:600;
     letter-spacing:1px;text-transform:uppercase;margin:0 0 8px">{algo}</p>
  <p style="color:#3a4d6b;font-size:11px;line-height:1.6;margin:0 0 12px">{desc}</p>
  <div style="display:flex;gap:16px">
    <div>
      <p style="color:#253555;font-size:9px;margin:0;font-weight:700;
         letter-spacing:1px;text-transform:uppercase">ACCURACY</p>
      <p style="color:{clr};font-size:22px;font-weight:800;margin:3px 0">{v['acc']}%</p>
    </div>
    <div>
      <p style="color:#253555;font-size:9px;margin:0;font-weight:700;
         letter-spacing:1px;text-transform:uppercase">ROC-AUC</p>
      <p style="color:{clr};font-size:22px;font-weight:800;margin:3px 0">{v['auc']}</p>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EXPLORE DATA
# ─────────────────────────────────────────────────────────────
elif cur == "Explore Data":
    ph("Data Explorer",
       f"{len(master):,} rows · {master.shape[1]} columns","🔍")
    col_sel = st.multiselect("Columns:",master.columns.tolist(),
        default=["gender","age_band","highest_education","final_result",
                 "total_clicks","avg_assessment_score","active_days"])
    c1,c2 = st.columns(2)
    n_rows = c1.slider("Rows",10,500,50,10)
    rf = c2.multiselect("Filter result:",master["final_result"].unique(),
                         default=list(master["final_result"].unique()))
    view = master[master["final_result"].isin(rf)]
    if col_sel: view = view[col_sel]
    st.dataframe(view.head(n_rows), use_container_width=True)
    r1,r2,r3 = st.columns(3)
    r1.metric("Total Rows",    f"{len(master):,}")
    r2.metric("Total Columns", master.shape[1])
    r3.metric("Missing Values",int(master.isnull().sum().sum()))

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#16213a;font-size:11px'>"
            "🎓 OULAD Analytics · Streamlit · Open University</p>",
            unsafe_allow_html=True)
