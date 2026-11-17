"""Disease Prediction System - training pipeline.

Usage:
    python train.py                       # uses data/Training.csv if present, else synthetic demo data
    python train.py --data data/Training.csv

Dataset format: one column per symptom (0/1), last column 'prognosis' (disease name).
Kaggle "Disease Prediction Using Machine Learning" (Training.csv / Testing.csv) works as-is.
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

SEED = 42
BASE_DIR = Path(__file__).resolve().parent

# ---------- demo data (used only if no real CSV is supplied) ----------
DISEASE_SYMPTOMS = {
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough", "mild_fever", "fatigue"],
    "Influenza": ["high_fever", "chills", "body_ache", "cough", "fatigue", "headache", "sore_throat"],
    "COVID-19": ["high_fever", "dry_cough", "loss_of_smell", "fatigue", "breathlessness", "body_ache"],
    "Malaria": ["high_fever", "chills", "sweating", "headache", "nausea", "vomiting", "body_ache"],
    "Dengue": ["high_fever", "severe_headache", "pain_behind_eyes", "joint_pain", "skin_rash", "nausea"],
    "Typhoid": ["high_fever", "abdominal_pain", "headache", "constipation", "fatigue", "loss_of_appetite"],
    "Gastroenteritis": ["diarrhoea", "vomiting", "nausea", "abdominal_pain", "mild_fever", "dehydration"],
    "Migraine": ["severe_headache", "nausea", "light_sensitivity", "vomiting", "visual_disturbance"],
    "Diabetes": ["excessive_thirst", "frequent_urination", "fatigue", "blurred_vision", "weight_loss", "slow_healing"],
    "Hypertension": ["headache", "dizziness", "chest_pain", "blurred_vision", "breathlessness"],
    "Asthma": ["breathlessness", "wheezing", "dry_cough", "chest_tightness", "fatigue"],
    "Pneumonia": ["high_fever", "cough", "chest_pain", "breathlessness", "chills", "fatigue"],
    "Jaundice": ["yellow_skin", "dark_urine", "abdominal_pain", "fatigue", "loss_of_appetite", "nausea"],
    "Anemia": ["fatigue", "pale_skin", "dizziness", "breathlessness", "headache"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "abdominal_pain", "mild_fever", "dark_urine"],
}

def make_synthetic(n_per_class=300, p_core=0.75, p_noise=0.04):
    rng = np.random.default_rng(SEED)
    symptoms = sorted({s for v in DISEASE_SYMPTOMS.values() for s in v})
    rows = []
    for disease, core in DISEASE_SYMPTOMS.items():
        for _ in range(n_per_class):
            r = {s: int(rng.random() < (p_core if s in core else p_noise)) for s in symptoms}
            r["prognosis"] = disease
            rows.append(r)
    return pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------- training ----------
def main(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    has_real_data = path.is_file()
    if has_real_data:
        df = pd.read_csv(path).dropna(axis=1, how="all")
        print(f"Loaded real dataset: {path} {df.shape}")
    else:
        df = make_synthetic()
        print(f"No dataset found -> using synthetic demo data {df.shape}")
    df = df.drop_duplicates() if has_real_data else df

    X = df.drop(columns="prognosis").astype(int)
    le = LabelEncoder(); y = le.fit_transform(df["prognosis"].str.strip())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=2.0),
        "BernoulliNB": BernoulliNB(),
        "SVC": SVC(kernel="rbf", C=5, probability=True, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    }
    print("\n5-fold CV (train set, macro-F1):")
    scores = {}
    for name, m in candidates.items():
        s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        scores[name] = s.mean(); print(f"  {name:20s} {s.mean():.4f} +/- {s.std():.4f}")

    # tune the random forest
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=SEED),
        {"n_estimators": [200, 400, 600], "max_depth": [None, 10, 20, 30],
         "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 0.3]},
        n_iter=12, cv=cv, scoring="f1_macro", random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    print(f"\nTuned RandomForest CV macro-F1: {search.best_score_:.4f}  params={search.best_params_}")
    candidates["TunedRandomForest"] = search.best_estimator_
    scores["TunedRandomForest"] = search.best_score_

    # soft-voting ensemble of the 3 best models
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    ens = VotingClassifier([(n, candidates[n]) for n in top3], voting="soft", n_jobs=-1)
    ens_cv = cross_val_score(ens, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    print(f"Soft-voting ensemble {top3}: {ens_cv:.4f}")
    candidates["Ensemble"] = ens; scores["Ensemble"] = ens_cv

    best_name = max(scores, key=scores.get)
    best = candidates[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
    print(f"\nBEST MODEL: {best_name}\nHeld-out test  accuracy={acc:.4f}  macro-F1={f1:.4f}\n")
    print(classification_report(y_te, pred, target_names=le.classes_, zero_division=0))

    model_dir = BASE_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"model": best, "symptoms": list(X.columns), "classes": list(le.classes_)},
                model_dir / "disease_model.joblib")
    report = {
        "best_model": best_name,
        "cv_scores": scores,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "dataset_source": f"CSV: {path.name}" if has_real_data else "Synthetic demo data",
        "dataset_rows": len(df),
        "test_rows": len(y_te),
        "symptom_count": X.shape[1],
        "disease_count": len(le.classes_),
    }
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues"); ax.set_title(f"Confusion matrix - {best_name}")
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=90, fontsize=7); ax.set_yticklabels(le.classes_, fontsize=7)
    plt.tight_layout(); plt.savefig(model_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    print("Saved model/disease_model.joblib, metrics.json, confusion_matrix.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/Training.csv")
    main(ap.parse_args().data)
