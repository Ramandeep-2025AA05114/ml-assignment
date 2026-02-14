import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# -----------------------------
# Load models and scaler ONCE
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler


models, scaler = load_models()

# -----------------------------
# UI
# -----------------------------
st.title("Mobile Price Classification App")

st.write(
    "This app loads pre-trained machine learning models and performs "
    "prediction and evaluation on a small uploaded test dataset."
)

st.sidebar.header("Upload Test Dataset & Select Model")

uploaded_file = st.sidebar.file_uploader(
    "Upload test.csv (must include price_range column)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

# -----------------------------
# Prediction Logic
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Test Dataset Preview")
    st.dataframe(df.head())

    if "price_range" not in df.columns:
        st.error("Uploaded file must contain 'price_range' column")
    else:
        X = df.drop("price_range", axis=1)
        y = df["price_range"]

        model = models[model_name]

        # Apply scaling ONLY for required models
        if model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob, multi_class="ovr")
        precision = precision_score(y, y_pred, average="weighted")
        recall = recall_score(y, y_pred, average="weighted")
        f1 = f1_score(y, y_pred, average="weighted")
        mcc = matthews_corrcoef(y, y_pred)

        # -----------------------------
        # Display Metrics
        # -----------------------------
        st.subheader("Evaluation Metrics")

        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**AUC:** {auc:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**MCC:** {mcc:.4f}")

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)