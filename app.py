"""Streamlit UI: run `python train.py` once, then `streamlit run app.py`."""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "disease_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "model" / "confusion_matrix.png"

st.set_page_config(
    page_title="Disease Prediction | ML Demo",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        st.error("The trained model is missing. Run `python train.py` in the project folder, then refresh this page.")
        st.stop()
    return joblib.load(MODEL_PATH)


def show_score_chart(data, alt_text, height, category_label, score_label):
    chart = (
        alt.Chart(data)
        .mark_bar(color="#2DD4BF", cornerRadiusEnd=4)
        .encode(
            x=alt.X("Score:Q", title=score_label, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Category:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("Category:N", title=category_label),
                alt.Tooltip("Score:Q", title=score_label, format=".1%"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, alt=alt_text, width="stretch")


bundle = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
model, symptoms, classes = bundle["model"], bundle["symptoms"], bundle["classes"]
pretty = {symptom: symptom.replace("_", " ").title() for symptom in symptoms}
is_synthetic_demo = metrics["dataset_source"] == "Synthetic demo data"

translations = {
    "English": {
        "title": "🩺 Disease Prediction System",
        "caption": "An interactive machine-learning project for learning and classroom demonstration.",
        "predict_tab": "🧪 Try a prediction",
        "insights_tab": "📊 Model insights",
        "about_tab": "ℹ️ How it works",
        "sample_heading": "Explore sample scenarios",
        "sample_caption": "Choose an example to populate symptoms, or select symptoms manually.",
        "sample_cold": "🤧 Cold-like sample",
        "sample_flu": "🌡️ Flu-like sample",
        "sample_headache": "🧠 Headache sample",
        "symptoms": "Symptoms",
        "symptoms_placeholder": "Search and select symptoms",
        "predict": "Predict",
        "select_symptom": "Select at least one symptom to try the demo.",
        "output_heading": "Demo model output",
        "top_result": "Top model result",
        "model_score": "Model score (not medical confidence)",
        "top_chart_alt": "Model scores for the top three disease categories",
        "category": "Category",
        "score": "Model score",
        "score_note": "These scores are model outputs, not a diagnosis or calibrated medical probabilities.",
        "low_score": "The top model score is low. This demo cannot determine a real health condition.",
        "synthetic_warning": (
            "This model uses synthetic demo data. Its results and test metrics do not represent real-world "
            "medical accuracy."
        ),
        "insights_heading": "Training snapshot",
        "insights_warning": "Synthetic demo data was used. These metrics are for demonstrating the ML workflow only.",
        "selected_model": "Selected model",
        "test_accuracy": "Demo test accuracy",
        "macro_f1": "Macro F1",
        "data_summary": "Data source: {source} · {rows:,} rows · {symptoms} symptoms · {classes} classes",
        "cv_heading": "Cross-validation model comparison",
        "cv_chart_alt": "Cross-validation macro F1 scores by model",
        "confusion_matrix": "Confusion matrix",
        "matrix_caption": "Held-out test results for the current training dataset.",
        "matrix_alt": "Confusion matrix showing actual and predicted disease categories",
        "matrix_missing": "Confusion matrix image is missing. Run `python train.py` to regenerate the model reports.",
        "about_heading": "How the demo works",
        "flow_symptoms": "Select symptoms",
        "flow_features": "Create binary features",
        "flow_model": "Trained ML model",
        "flow_results": "Top 3 model scores",
        "flow_alt": "Flow from symptom selection through feature preparation and the trained model to three scores",
        "step_symptoms": "**1. Select symptoms**\n\nChoose symptoms or load a sample scenario.",
        "step_features": "**2. Prepare features**\n\nSelected symptoms are represented as binary model inputs.",
        "step_model": "**3. Compare model scores**\n\nThe trained classifier ranks three output categories.",
        "disclaimer": (
            "This is a classroom machine-learning demonstration, not a medical device. "
            "Do not use it to diagnose, treat, or make health decisions. Consult a qualified doctor."
        ),
        "data_note": (
            "Current data source: {source}. Replace synthetic demo data with a properly sourced dataset for research."
        ),
        "synthetic_source": "Synthetic demo data",
        "disease_names": {},
    },
}

t = translations["English"]
display_source = t["synthetic_source"] if is_synthetic_demo else metrics["dataset_source"]
symptom_labels = pretty

st.title(t["title"])
st.caption(t["caption"])

predict_tab, insights_tab, about_tab = st.tabs(
    [t["predict_tab"], t["insights_tab"], t["about_tab"]]
)

with predict_tab:
    st.subheader(t["sample_heading"])
    st.caption(t["sample_caption"])

    sample_scenarios = [
        ("cold", t["sample_cold"], ["runny_nose", "sneezing", "sore_throat", "cough"]),
        ("flu", t["sample_flu"], ["high_fever", "chills", "body_ache", "cough", "fatigue"]),
        ("headache", t["sample_headache"], ["severe_headache", "nausea", "light_sensitivity"]),
    ]

    for column, (key, label, sample_symptoms) in zip(st.columns(3), sample_scenarios):
        if column.button(label, key=f"sample_{key}", width="stretch"):
            st.session_state["selected_symptoms"] = sample_symptoms
            st.rerun()

    chosen = st.multiselect(
        t["symptoms"],
        symptoms,
        format_func=symptom_labels.get,
        key="selected_symptoms",
        placeholder=t["symptoms_placeholder"],
    )

    if st.button(t["predict"], type="primary"):
        if not chosen:
            st.warning(t["select_symptom"])
        else:
            features = pd.DataFrame([[int(symptom in chosen) for symptom in symptoms]], columns=symptoms)
            probabilities = model.predict_proba(features)[0]
            top = pd.Series(probabilities, index=classes).sort_values(ascending=False).head(3)

            st.markdown(f"#### {t['output_heading']}")
            result_col, score_col = st.columns(2)
            disease_name = t["disease_names"].get(top.index[0], top.index[0])
            result_col.metric(t["top_result"], disease_name, border=True)
            score_col.metric(t["model_score"], f"{top.iloc[0]:.0%}", border=True)

            chart_data = top.rename_axis("Category").rename("Score").reset_index()
            chart_data["Category"] = chart_data["Category"].map(
                lambda category: t["disease_names"].get(category, category)
            )
            show_score_chart(
                chart_data,
                t["top_chart_alt"],
                height=150,
                category_label=t["category"],
                score_label=t["score"],
            )
            st.caption(t["score_note"])
            if top.iloc[0] < 0.5:
                st.info(t["low_score"])

    if is_synthetic_demo:
        st.warning(t["synthetic_warning"])

with insights_tab:
    st.subheader(t["insights_heading"])
    if is_synthetic_demo:
        st.warning(t["insights_warning"])

    metric_columns = st.columns(3)
    metric_columns[0].metric(t["selected_model"], metrics["best_model"], border=True)
    metric_columns[1].metric(t["test_accuracy"], f"{metrics['test_accuracy']:.1%}", border=True)
    metric_columns[2].metric(t["macro_f1"], f"{metrics['test_macro_f1']:.1%}", border=True)

    st.caption(
        t["data_summary"].format(
            source=display_source,
            rows=metrics["dataset_rows"],
            symptoms=metrics["symptom_count"],
            classes=metrics["disease_count"],
        )
    )

    st.markdown(f"#### {t['cv_heading']}")
    score_data = pd.DataFrame(
        [{"Category": name, "Score": score} for name, score in metrics["cv_scores"].items()]
    ).sort_values("Score", ascending=False)
    show_score_chart(
        score_data,
        t["cv_chart_alt"],
        height=250,
        category_label=t["category"],
        score_label=t["macro_f1"],
    )

    if CONFUSION_MATRIX_PATH.is_file():
        st.markdown(f"#### {t['confusion_matrix']}")
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption=t["matrix_caption"],
            alt=t["matrix_alt"],
        )
    else:
        st.error(t["matrix_missing"])

with about_tab:
    st.subheader(t["about_heading"])
    flow = f"""flowchart LR
        A["{t['flow_symptoms']}"] --> B["{t['flow_features']}"]
        B --> C["{t['flow_model']}"]
        C --> D["{t['flow_results']}"]
    """
    st.mermaid_chart(flow, alt=t["flow_alt"])

    steps = st.columns(3)
    steps[0].markdown(t["step_symptoms"])
    steps[1].markdown(t["step_features"])
    steps[2].markdown(t["step_model"])

    st.info(t["disclaimer"])
    st.caption(t["data_note"].format(source=display_source))
