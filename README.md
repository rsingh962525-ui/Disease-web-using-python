# Disease Prediction System (Python + ML)

Predicts a likely disease from a set of symptoms (multi-class classification).

## Pipeline
1. Load symptom matrix (binary features) - real CSV or built-in synthetic demo data
2. Stratified 80/20 split, 5-fold CV comparison of 5 models (LogReg, BernoulliNB, SVC, RandomForest, ExtraTrees)
3. RandomizedSearchCV tuning of Random Forest
4. Soft-voting ensemble of the top 3 models
5. Best model (by CV macro-F1) evaluated on held-out test set; saved with joblib
6. Streamlit app returns top-3 diseases with probabilities

## Run on Windows
After the one-time setup below, double-click `Open Disease Prediction.vbs` to start
the app and open it in your browser without using a terminal.

For first-time setup, open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at `http://localhost:8501`. The model is saved under `model/`, so the
app also works when launched from a different current directory.

The presentation UI includes a black theme, English-only labels, sample symptom
scenarios, a workflow diagram, cross-validation model comparison, and a confusion
matrix. When using the built-in synthetic dataset, the app labels its metrics as
demonstration-only.

## Use a real dataset (recommended)
Download Kaggle's "Disease Prediction Using Machine Learning" (`Training.csv`, 132 symptoms, 41 diseases)
into `data/Training.csv`, then run `python train.py`. Any CSV with 0/1 symptom columns and a
`prognosis` label column works.

## Notes
- Built-in demo data is synthetic, so its accuracy says little about real-world performance.
- Not a medical device; for learning purposes only.
