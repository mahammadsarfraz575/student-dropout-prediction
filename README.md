# 🎓 Student Dropout Prediction — End-to-End ML Pipeline

> Predict university student dropout risk with **88% ROC-AUC** using engagement data  
> **32,593 students** · **7 relational tables** · **30+ engineered features** · **100% automated**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 Live Demo

---

## 📊 Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **ROC-AUC** | 0.88 | Model separates Pass/Fail extremely well |
| **Accuracy** | 85% | Overall correct predictions |
| **Early Warning** | Day 30 | Flag at-risk students before dropout |
| **Engagement Gap** | 3× | Passing students click 3× more than failing |
| **Deprivation Gap** | 15pt | Pass rate gap between rich/poor students |

---

## 🎯 Business Impact

**Problem:** Universities lose revenue when students withdraw early.

**Solution:** Machine learning system that flags at-risk students **by day 30** — before they drop out — with 85%+ accuracy.

**Value:** Early intervention saves institutions ~$5,000 per retained student.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost, Imbalanced-learn |
| **Dashboard** | Streamlit |
| **BI Integration** | Power BI Desktop + Power BI Service |
| **Data Source** | Open University Learning Analytics Dataset (OULAD) |
| **Deployment** | Streamlit Cloud (free tier) |

---

## ⚡ Quick Start (3 Commands)

```bash
# 1. Clone repo
git clone https://github.com/yourname/oulad-project.git
cd oulad-project

# 2. Install dependencies + download data (NO API KEY NEEDED!)
pip install -r requirements.txt
python 00_setup_github.py

# 3. Run full pipeline (15 minutes)
python run_all.py

# 4. Launch dashboard
streamlit run 04_dashboard.py
```

**That's it!** Your dashboard is live at `http://localhost:8501`

---

## 📂 Project Structure

```
OULAD_Project/
├── 📄 README.md                    ← You are here
├── 📘 PIPELINE_EXPLAINED.md        ← Full technical explanation
├── 📗 INTERVIEW_GUIDE.md           ← Algorithm deep-dive + Q&A
├── 📙 GITHUB_SETUP.md              ← Deployment guide
│
├── 🐍 00_setup_github.py          ← Setup (NO API key needed!)
├── 🐍 01_eda.py                    ← Exploratory Data Analysis
├── 🐍 02_feature_engineering.py   ← Build 30+ ML features
├── 🐍 03_ml_models.py              ← Train 4 models, evaluate
├── 🐍 04_dashboard.py              ← Streamlit app (6 tabs)
├── 🐍 05_powerbi_export.py         ← Generate 10-sheet Excel
├── 🐍 run_all.py                   ← Master runner
│
├── 📦 requirements.txt
├── 🔧 .gitignore
│
├── 📁 data/                        ← 7 CSV files (auto-downloaded)
├── 📁 outputs/                     ← Charts, reports, CSVs
├── 📁 models/                      ← Trained .pkl files
└── 📁 exports/                     ← Power BI Excel file
```

---

## 📊 The 7 Data Tables

| Table | Rows | Description |
|-------|------|-------------|
| `studentInfo.csv` | 32,593 | Demographics (age, gender, education, deprivation index) |
| `studentVle.csv` | 10,655,280 | Every click a student made on learning materials |
| `studentAssessment.csv` | 173,912 | Grades for each assignment |
| `assessments.csv` | 206 | Assignment types, weights, due dates |
| `studentRegistration.csv` | 32,593 | Enrollment + withdrawal dates |
| `courses.csv` | 22 | Module codes and presentation dates |
| `vle.csv` | 6,364 | Types of learning materials |

**Challenge:** Merge all 7 tables → one master dataset with 30+ features per student.

---

## 🔬 Feature Engineering (30+ Features)

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| `total_clicks` | Total VLE engagement | Top predictor — 3× difference Pass vs Fail |
| `clicks_day30` | Early engagement | **Early warning signal** — predicts outcome by day 30 |
| `active_days` | Consistency | Regular engagement = higher success |
| `avg_assessment_score` | Academic performance | Direct ability indicator |
| `tma_avg_score` | Coursework average | Measures sustained effort |
| `exam_avg_score` | Final exam | High-stakes performance |
| `late_submissions` | Time management | Behavioral proxy |
| `withdrawal_day` | Timing of dropout | Early vs late withdrawal patterns |
| `imd_band` (encoded) | Socioeconomic status | 15-point pass rate gap |

---

## 🤖 ML Models Trained

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| **Logistic Regression** | 79% | 0.82 | Baseline — fast, interpretable |
| **Random Forest** | 83% | 0.86 | Feature importance analysis |
| **Gradient Boosting** | 84% | 0.87 | Sequential error correction |
| **XGBoost** ⭐ | **85%** | **0.88** | Best performer — production model |

**Why XGBoost won:**
- Handles imbalanced classes well
- Built-in regularization prevents overfitting
- Industry standard for tabular data

---

## 📈 Key Findings

### 1️⃣ **Engagement is Everything**
Students who **pass** average **4,200 VLE clicks**.  
Students who **fail** average **1,800 clicks**.  
Students who **withdraw** average **800 clicks**.

→ **3× engagement gap** between success and failure.

### 2️⃣ **Early Warning Window Exists**
By **day 30**, students with **<200 clicks** have a **70% dropout rate**.  
→ Intervention at day 30 can save students before it's too late.

### 3️⃣ **Socioeconomic Gap is Real**
Students from **most deprived areas** (IMD 0-10%): **62% pass rate**  
Students from **least deprived areas** (IMD 90-100%): **77% pass rate**  
→ **15-point gap** — education inequality is measurable.

---

## 📊 Streamlit Dashboard (6 Tabs)

| Tab | Features |
|-----|----------|
| **📊 Overview** | Result distribution, pass rates by module |
| **🔍 Demographics** | Gender, age, education, deprivation analysis |
| **📈 Engagement** | VLE clicks over time, engagement vs outcome |
| **📝 Assessments** | Score distributions, TMA vs Exam comparison |
| **🤖 ML Prediction** | Live risk predictor — input student data → get risk score |
| **💡 Insights** | Key findings + ready-to-post LinkedIn summary |

---

## 📊 Power BI Integration

The pipeline auto-generates a **10-sheet Excel file** ready for Power BI:

1. `Master_Data` — All features for filtering
2. `Result_Summary` — By demographics
3. `Module_Performance` — Pass rates by course
4. `Engagement_Bands` — Low/Medium/High engagement
5. `IMD_Deprivation` — Socioeconomic analysis
6. `Daily_VLE_Activity` — Time series
7. `Assessment_Performance` — TMA vs Exam
8. `Early_Warning_Day30` — Risk flags
9. `Withdrawal_Timeline` — Dropout timing
10. `KPI_Summary` — Top-level metrics

**See:** `06_powerbi_publish.md` for how to embed Power BI in Streamlit.

---

## 🎓 What This Project Demonstrates

✅ **End-to-end ML pipeline** — data acquisition → deployment  
✅ **Production-grade code** — modular, version-controlled, documented  
✅ **Multi-table data engineering** — 7 tables → 1 master dataset  
✅ **Advanced feature engineering** — 30+ derived features  
✅ **Model comparison** — 4 algorithms benchmarked  
✅ **Interactive deployment** — Streamlit dashboard  
✅ **BI integration** — Power BI for stakeholders  
✅ **Business impact** — Early intervention system  

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `PIPELINE_EXPLAINED.md` | Full technical breakdown of every step |
| `INTERVIEW_GUIDE.md` | Algorithm explanations + top 10 interview Q&A |
| `GITHUB_SETUP.md` | Deployment guide (Streamlit Cloud + Power BI) |

---

## 🎤 Interview-Ready

This project is designed to be **explained in interviews**:

- **30-second pitch:** ✅ Covered in opening section
- **Technical deep-dive:** ✅ See `PIPELINE_EXPLAINED.md`
- **Algorithm explanations:** ✅ See `INTERVIEW_GUIDE.md`
- **Business value:** ✅ Early intervention saves $5K/student
- **Reproducible:** ✅ Clone repo → run → done

---

## 📧 Contact

**Mahammad Sarfraz**  
📧 mahammadsarfraz575@gmail.com  
---

## 📄 License

MIT License — free to use for learning and portfolios.

---

## 🙏 Acknowledgments

- **Data Source:** [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
- **Inspiration:** Real-world need for early intervention in education

---

**⭐ If this helped you, please star the repo!**
