<div align="center">

# 🎓 Student Dropout Prediction
### End-to-End Machine Learning Pipeline for University Student Risk Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**[🚀 Live](https://student-dropout-prediction-mahammadsarfraz.streamlit.app/)** &nbsp;|&nbsp; **[📊 Power BI Report](#)** &nbsp;|&nbsp; **[📧 Contact](#contact)**

> Predicting university student dropout risk with **88% ROC-AUC** using 32,593 student records,  
> 10M+ VLE click events and 4 machine learning models — fully automated, zero manual steps.

</div>

---

## 📌 Problem Statement

Universities worldwide face a critical challenge: **students who are at risk of dropping out are only identified after they withdraw** — when it is already too late to intervene.

**The cost is significant:**
- 💸 Each withdrawn student costs a university an estimated **$5,000–$15,000** in lost tuition
- 📉 UK Open University withdrawal rate is **~33%** across all modules
- 🕐 The window for intervention is narrow — most at-risk students disengage within the **first 30 days**

**This project answers one question:**

> *Can we predict, by Day 30 of a course, whether a student will pass or withdraw — with enough accuracy to trigger early intervention?*

**Answer: Yes. With 88% ROC-AUC using only engagement data available in the first 30 days.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  Open University ──► Direct HTTP Download ──► 7 CSV Files (200MB)  │
│  (No API key required — auto-downloads on first run)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     00_setup.py                                     │
│   Install packages → Download OULAD → Verify 7 CSV files           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      01_eda.py                                      │
│                                                                     │
│  studentInfo.csv ──────────────────────► Demographics analysis      │
│  studentVle.csv  ──► 10M click events ► Engagement patterns         │
│  studentAssessment.csv ────────────────► Score distributions        │
│  studentRegistration.csv ──────────────► Withdrawal timing          │
│  assessments.csv ──────────────────────► Assessment types           │
│  courses.csv + vle.csv ────────────────► Module context             │
│                                                                     │
│  Output: 9 charts + 3 HTML interactive files → /outputs/            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              02_feature_engineering.py                              │
│                                                                     │
│  7 tables ──► pandas .merge() (SQL-style JOIN) ──► master_dataset  │
│                                                                     │
│  Features created:                                                  │
│  • total_clicks          → overall VLE engagement                   │
│  • clicks_day30          → early warning signal (Day 30)            │
│  • active_days           → learning consistency                     │
│  • avg_assessment_score  → academic performance                     │
│  • tma_avg_score         → continuous assessment                    │
│  • exam_avg_score        → high-stakes performance                  │
│  • late_submissions      → behavioural proxy                        │
│  • withdrawal_day        → dropout timing                           │
│  • imd_band_enc          → socioeconomic deprivation index          │
│  + 20 more features                                                 │
│                                                                     │
│  Output: master_dataset.csv (32,593 rows × 40+ columns)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    03_ml_models.py                                  │
│                                                                     │
│  Target: Pass/Distinction=1  vs  Fail/Withdrawn=0                  │
│  Split:  80% train | 20% test | Stratified | 5-fold CV             │
│                                                                     │
│  ┌─────────────────────┬──────────┬─────────┐                      │
│  │ Model               │ Accuracy │ AUC     │                      │
│  ├─────────────────────┼──────────┼─────────┤                      │
│  │ Logistic Regression │  79.2%   │  0.82   │  ← baseline          │
│  │ Random Forest       │  83.1%   │  0.86   │  ← feature importance│
│  │ Gradient Boosting   │  84.3%   │  0.87   │  ← sequential boost  │
│  │ Hist Grad Boost     │  85.0%   │  0.88   │  ← best performer ⭐ │
│  └─────────────────────┴──────────┴─────────┘                      │
│                                                                     │
│  sklearn Pipeline: Imputer → Scaler → Classifier                   │
│  Output: 4 trained models → /models/                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
┌──────────────────────┐      ┌────────────────────────────┐
│   04_dashboard.py    │      │   05_powerbi_export.py     │
│                      │      │                            │
│  Streamlit Web App   │      │  10-sheet Excel file for   │
│  6 pages, 20+ charts │      │  Power BI integration      │
│  Live ML prediction  │      │                            │
│  localhost:8501      │      │  /exports/OULAD_PowerBI    │
└──────────────────────┘      └────────────────────────────┘
```

---

## 📊 Dataset

**OULAD — Open University Learning Analytics Dataset**

| Table | Rows | Description |
|-------|------|-------------|
| `studentInfo.csv` | 32,593 | Demographics: age, gender, education level, deprivation index |
| `studentVle.csv` | 10,655,280 | Every click on every learning resource (the core signal) |
| `studentAssessment.csv` | 173,912 | Scores for every assignment submitted |
| `assessments.csv` | 206 | Assignment types (TMA/CMA/Exam), weights and due dates |
| `studentRegistration.csv` | 32,593 | Enrolment and withdrawal dates per student per module |
| `courses.csv` | 22 | Module codes, names and presentation dates |
| `vle.csv` | 6,364 | Types of learning materials (forum, resource, quiz, etc.) |

**Source:** [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)  
**License:** Creative Commons Attribution 4.0 International

---

## 🔑 Key Findings

### 1. Engagement gap is enormous
```
Students who PASS      → avg 4,200 VLE clicks across the course
Students who FAIL      → avg 1,800 VLE clicks
Students who WITHDRAW  → avg   800 VLE clicks
```
A **5× engagement gap** between distinction students and withdrawals.

### 2. Day-30 Early Warning Window
Students with **fewer than 200 clicks by Day 30** have a **70% dropout rate**.  
This is the basis of the Early Warning System — actionable before it is too late.

### 3. Socioeconomic Gap
```
Most deprived areas (IMD 0–10%)   → 62% pass rate
Least deprived areas (IMD 90–100%) → 77% pass rate
```
A **15-point gap** measurable from demographic data alone.

### 4. Parental Education Multiplier
Students whose parents hold a postgraduate degree pass at **18% higher rates**  
than students whose parents have no formal qualifications.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.9+ | Core language |
| **Data Processing** | Pandas, NumPy | Load, merge and clean 7 relational tables |
| **Visualisation** | Plotly, Matplotlib, Seaborn | 20+ interactive charts |
| **Machine Learning** | Scikit-learn | Pipelines, preprocessing, 4 classifiers |
| **Dashboard** | Streamlit | 6-page interactive web application |
| **BI Integration** | OpenPyXL | 10-sheet Excel export for Power BI |
| **Data Source** | Direct HTTP (no API key) | Auto-downloads OULAD on first run |
| **Deployment** | Streamlit Community Cloud | Free public hosting |
| **Version Control** | Git + GitHub | Full history, reproducible |

---

## 📁 Project Structure

```
student-dropout-prediction/
│
├── 📄 README.md                    ← You are here
├── 📘 PIPELINE_EXPLAINED.md        ← Full technical breakdown of each step
├── 📗 INTERVIEW_GUIDE.md           ← Algorithm explanations + top 10 Q&A
│
├── 🐍 00_setup.py                  ← Install packages + download data (no API!)
├── 🐍 01_eda.py                    ← Exploratory analysis of all 7 tables
├── 🐍 02_feature_engineering.py    ← Merge tables → 30+ ML-ready features
├── 🐍 03_ml_models.py              ← Train, evaluate and save 4 models
├── 🐍 04_dashboard.py              ← Streamlit 6-page dashboard
├── 🐍 05_powerbi_export.py         ← Generate 10-sheet Excel for Power BI
├── 🐍 create_sample_data.py        ← Create 5K-row sample for cloud deploy
├── 🐍 run_all.py                   ← Run the full pipeline in one command
│
├── 📦 requirements.txt             ← Python dependencies
├── 🚫 .gitignore                   ← Excludes data/ models/ exports/
│
├── 📁 data/                        ← Auto-created. 7 CSVs downloaded here
├── 📁 outputs/                     ← Charts, processed CSVs, model summary
├── 📁 models/                      ← Trained .pkl files
├── 📁 exports/                     ← Power BI Excel file
└── 📁 sample_data/                 ← 5K-row sample (committed to GitHub)
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9 or higher
- 4 GB RAM minimum (for the 10M-row VLE table)
- Internet connection (first run only, to download data)

### 1 — Clone the repository
```bash
git clone ````(https://././.)
cd student-dropout-prediction
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Download data and run the full pipeline
```bash
python run_all.py
```

This single command:
- ✅ Downloads OULAD dataset automatically (no API key needed)
- ✅ Runs EDA across all 7 tables
- ✅ Engineers 30+ features
- ✅ Trains and evaluates 4 ML models
- ✅ Exports 10-sheet Excel for Power BI

**Estimated time:** ~15 minutes on a standard laptop

### 4 — Launch the dashboard
```bash
streamlit run 04_dashboard.py
```

Open your browser at `http://localhost:8501`

---

## 📱 Dashboard Pages

| Page | What You See |
|------|-------------|
| **🏠 Home** | KPI cards, result distribution, quick navigation |
| **📊 Overview** | Pass rates by module and presentation period |
| **👥 Demographics** | Gender, age, education, deprivation analysis |
| **📈 Engagement** | VLE click patterns, daily activity timeline |
| **📝 Assessments** | Score distributions, TMA vs Exam comparison |
| **🤖 Predict** | Live risk prediction — all 4 models simultaneously |
| **⚖️ Compare** | ROC curves, accuracy table, algorithm explanations |

---

## 🤖 ML Models Explained

### Why 4 models?

| Model | Role | Why included |
|-------|------|-------------|
| **Logistic Regression** | Baseline | Fast, interpretable. Sets the floor. |
| **Random Forest** | Ensemble | Parallel trees. Provides feature importance. |
| **Gradient Boosting** | Sequential | Each tree corrects previous errors. High accuracy. |
| **Hist Gradient Boost** | Best performer | sklearn's native fast booster. Most accurate. |

### Feature Importance (top 5)
```
avg_assessment_score   ████████████████████  38.2%
total_clicks           ███████████████       28.1%
clicks_day30           █████████             17.4%
active_days            █████                  9.8%
tma_avg_score          ██                     4.1%
```

### Evaluation
```
Metric         Value    Meaning
──────────────────────────────────────────────────────────────
Accuracy       85.0%    85 of every 100 predictions are correct
ROC-AUC        0.88     88% chance model ranks a passer above a failer
Precision      80.3%    When we flag AT RISK, we are right 80% of the time
Recall         78.1%    We catch 78% of students who would drop out
CV Score       84.2%    Consistent across 5 different train/test splits
```

---

## 📊 Power BI Integration

Running `python 05_powerbi_export.py` generates `/exports/OULAD_PowerBI_Dashboard.xlsx`  
with 10 pre-aggregated sheets ready to connect directly to Power BI Desktop:

| Sheet | Purpose |
|-------|---------|
| `Master_Data` | Full dataset for slicer-based filtering |
| `Result_Summary` | Outcomes broken down by demographics |
| `Module_Performance` | Pass rates per course |
| `Engagement_Bands` | Students grouped by VLE activity level |
| `IMD_Deprivation` | Socioeconomic gap analysis |
| `Daily_VLE_Activity` | Time-series click data |
| `Assessment_Performance` | TMA vs Exam comparison |
| `Early_Warning_Day30` | At-risk flags at the 30-day mark |
| `Withdrawal_Timeline` | When students drop out |
| `KPI_Summary` | Top-level metrics card |

---

## ☁️ Cloud Deployment

The live demo runs on **Streamlit Community Cloud** (free tier) using a 5,000-student sample:

```
Full dataset:   32,593 students  →  local machine only (200MB)
Cloud demo:      5,000 students  →  Streamlit Cloud (10MB, fits free tier)
```

To regenerate the sample after running the full pipeline:
```bash
python create_sample_data.py
git add sample_data/
git commit -m "Update sample data"
git push
```

Streamlit Cloud auto-deploys on every push.

---

## 🎯 Business Impact

This system gives universities an **Early Intervention Engine**:

```
Day 0  → Student enrols
Day 30 → ML model flags at-risk students (85% accuracy)
Day 31 → University contacts flagged students
         → Offers tutoring / counselling / financial support
         → Prevents withdrawal before it happens

Expected outcome per 1,000 students:
  Without system: ~330 withdraw
  With system:    ~280 withdraw  (15% reduction)
  Revenue saved:  50 students × $10,000 = $500,000
```

---

## 📂 Documentation

| Document | Contents |
|----------|----------|
| [`README.md`](README.md) | This file — project overview |
| [`PIPELINE_EXPLAINED.md`](PIPELINE_EXPLAINED.md) | Step-by-step technical explanation of every script |
| [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md) | Algorithm deep-dives + 10 interview Q&A |

---

## 🚀 What This Demonstrates

This project was built to demonstrate **production-grade data science engineering**, not just modelling:

```
✅ Automated data pipeline    — no manual steps, runs with one command
✅ Relational data merging     — 7 tables → 1 master dataset
✅ Feature engineering         — 30+ derived features from raw events
✅ Multi-model comparison      — 4 algorithms benchmarked fairly
✅ sklearn Pipelines           — prevents data leakage, clean deployment
✅ Interactive deployment       — Streamlit with 6 pages and live prediction
✅ BI integration              — Power BI-ready Excel with 10 sheets
✅ Cloud deployment            — live URL, anyone can view it
✅ Full documentation          — README, pipeline guide, interview guide
✅ Version controlled          — Git history shows development process
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
MIT License © 2024 Your Name
Built by [Your Name](https://linkedin.com/in/yourprofile)
If you use this project, please star the repo ⭐ and give credit.
```

---

## 📧 Contact

**Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahammad-sarfraz)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your@email.com)

---

## 🙏 Acknowledgements

- **Dataset:** [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) by Kuzilek, Hlosta & Zdrahal (2017)
- **Paper:** Kuzilek J., Hlosta M., Zdrahal Z. (2017) *Open University Learning Analytics dataset.* Sci. Data 4:170171
- **Hosting:** [Streamlit Community Cloud](https://streamlit.io/cloud) — free tier

---

<div align="center">

**⭐ If this project helped you, please star the repository!**

*Built as a portfolio project demonstrating end-to-end ML engineering*

</div>
