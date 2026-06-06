"""
╔══════════════════════════════════════════════════════════════╗
║   STEP 3 — ML Models: Train, Evaluate, Save                  ║
║   Models: Logistic, Random Forest, XGBoost                   ║
║   Command: python 03_ml_models.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd, numpy as np, os, warnings, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline        import Pipeline
from sklearn.impute           import SimpleImputer
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import OUTPUT_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")
def ok(m):     print(f"  OK  {m}")

# ─── LOAD DATA ─────────────────────────────────────────────────
banner("Loading master dataset")
master   = pd.read_csv(os.path.join(OUTPUT_DIR,"master_dataset.csv"))
features = pd.read_csv(os.path.join(OUTPUT_DIR,"feature_cols.csv"),
                        header=None)[0].tolist()
features = [f for f in features if f in master.columns]

X = master[features].fillna(0)
y = master["target"].fillna(0).astype(int)

print(f"  Samples: {len(X):,}  Features: {len(features)}")
print(f"  Class balance: {y.value_counts().to_dict()}")

X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ─── DEFINE MODELS ─────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf",     RandomForestClassifier(n_estimators=200, max_depth=12,
                                            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf",     GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                learning_rate=0.05, random_state=RANDOM_STATE))
    ]),
}
if HAS_XGB:
    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf",     XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   use_label_encoder=False, eval_metric="logloss",
                                   random_state=RANDOM_STATE, n_jobs=-1))
    ])

# ─── TRAIN & EVALUATE ──────────────────────────────────────────
banner("Training models")
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fig,axes = plt.subplots(2, len(models), figsize=(5*len(models), 8))
fig.suptitle("OULAD — ML Model Evaluation", fontsize=14, fontweight="bold")

for i,(name,pipe) in enumerate(models.items()):
    print(f"\n  Training: {name}")
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:,1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    cv_acc  = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()

    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    CV Acc   : {cv_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Fail/Withdraw","Pass"]))

    results[name] = {"accuracy":acc,"auc":auc,"cv_acc":cv_acc,
                      "y_pred":y_pred,"y_prob":y_prob}

    # Save model
    mpath = os.path.join(MODEL_DIR, f"{name.replace(' ','_').lower()}.pkl")
    joblib.dump(pipe, mpath)
    ok(f"Saved: {mpath}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[0,i]
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"])
    ax.set_title(f"{name}\nAcc={acc*100:.1f}%  AUC={auc:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ROC curve
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    ax2 = axes[1,i]
    ax2.plot(fpr,tpr,color="#4f8ef7",linewidth=2,label=f"AUC={auc:.3f}")
    ax2.plot([0,1],[0,1],"k--",alpha=0.4)
    ax2.set_title(f"ROC — {name}")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()

plt.tight_layout()
out = os.path.join(OUTPUT_DIR,"03_model_evaluation.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
ok(f"Saved: {out}")

# ─── FEATURE IMPORTANCE (best model) ──────────────────────────
banner("Feature Importance")
best_name = max(results, key=lambda n: results[n]["auc"])
print(f"  Best model by AUC: {best_name}")
best_pipe = models[best_name]

try:
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf,"feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
        print("\n  Top 15 features:")
        print(imp.head(15).round(4).to_string())

        fig2,ax=plt.subplots(figsize=(10,6))
        imp.head(20).sort_values().plot(kind="barh",ax=ax,color="#4f8ef7",alpha=0.85)
        ax.set_title(f"Top 20 Feature Importances — {best_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        out2 = os.path.join(OUTPUT_DIR,"03_feature_importance.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
        ok(f"Saved: {out2}")
except Exception as e:
    print(f"  Could not plot importance: {e}")

# ─── SUMMARY TABLE ────────────────────────────────────────────
banner("Results Summary")
summary = pd.DataFrame({
    n: {"Accuracy":f"{v['accuracy']*100:.2f}%",
        "ROC-AUC":f"{v['auc']:.4f}",
        "CV Accuracy":f"{v['cv_acc']*100:.2f}%"}
    for n,v in results.items()
}).T
print(summary.to_string())
summary.to_csv(os.path.join(OUTPUT_DIR,"03_model_summary.csv"))
ok("Saved: 03_model_summary.csv")

# Save best model name for dashboard
with open(os.path.join(MODEL_DIR,"best_model.txt"),"w") as f:
    f.write(best_name.replace(" ","_").lower())
ok(f"Best model saved: {best_name}")

banner("ML COMPLETE -> next: streamlit run 04_dashboard.py")
