import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = "model_artifacts"

# --- Load model artifacts once, cached across reruns (Streamlit reruns the whole script on every interaction) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/churn_rf_model.pkl")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
    feature_columns = joblib.load(f"{ARTIFACT_DIR}/model_features.pkl")
    numeric_cols = joblib.load(f"{ARTIFACT_DIR}/numeric_cols.pkl")
    return model, scaler, feature_columns, numeric_cols

model, scaler, feature_columns, numeric_cols = load_artifacts()


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Turns the raw form inputs into the exact one-hot/engineered row the model expects."""
    row = {
        "gender": 1 if inputs["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if inputs["senior_citizen"] == "Yes" else 0,
        "Partner": 1 if inputs["partner"] == "Yes" else 0,
        "Dependents": 1 if inputs["dependents"] == "Yes" else 0,
        "tenure": inputs["tenure"],
        "PhoneService": 1 if inputs["phone_service"] == "Yes" else 0,
        "MultipleLines": 1 if inputs["multiple_lines"] == "Yes" else 0,
        "OnlineSecurity": 1 if inputs["online_security"] == "Yes" else 0,
        "OnlineBackup": 1 if inputs["online_backup"] == "Yes" else 0,
        "DeviceProtection": 1 if inputs["device_protection"] == "Yes" else 0,
        "TechSupport": 1 if inputs["tech_support"] == "Yes" else 0,
        "StreamingTV": 1 if inputs["streaming_tv"] == "Yes" else 0,
        "StreamingMovies": 1 if inputs["streaming_movies"] == "Yes" else 0,
        "PaperlessBilling": 1 if inputs["paperless_billing"] == "Yes" else 0,
        "MonthlyCharges": inputs["monthly_charges"],
        "TotalCharges": inputs["total_charges"],
        "InternetService_DSL": 1 if inputs["internet_service"] == "DSL" else 0,
        "InternetService_Fiber optic": 1 if inputs["internet_service"] == "Fiber optic" else 0,
        "Contract_One year": 1 if inputs["contract"] == "One year" else 0,
        "Contract_Two year": 1 if inputs["contract"] == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if inputs["payment_method"] == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if inputs["payment_method"] == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if inputs["payment_method"] == "Electronic check" else 0,
    }

    # Engineered features - must match Phase 6 of the training notebook exactly
    row["AvgMonthlySpend"] = inputs["total_charges"] / (inputs["tenure"] + 1)

    tenure = inputs["tenure"]
    row["TenureBucket_12-24"] = 1 if 12 < tenure <= 24 else 0
    row["TenureBucket_24-48"] = 1 if 24 < tenure <= 48 else 0
    row["TenureBucket_48+"] = 1 if tenure > 48 else 0

    row["Fiber_x_MonthToMonth"] = 1 if (
        inputs["internet_service"] == "Fiber optic" and inputs["contract"] == "Month-to-month"
    ) else 0

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict(inputs: dict, threshold: float = 0.5) -> dict:
    row = build_feature_row(inputs)
    row[numeric_cols] = scaler.transform(row[numeric_cols])
    proba = model.predict_proba(row)[:, 1][0]
    prediction = int(proba >= threshold)
    return {"prediction": prediction, "probability": float(proba)}


# ------------------------- UI -------------------------

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their likelihood of churning.")

with st.form("customer_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.subheader("Account")
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.subheader("Services")
    col5, col6 = st.columns(2)
    with col5:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
    with col6:
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    threshold = st.slider(
        "Decision threshold (lower = flags more customers as at-risk)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    inputs = {
        "gender": gender, "senior_citizen": senior_citizen, "partner": partner,
        "dependents": dependents, "tenure": tenure, "phone_service": phone_service,
        "multiple_lines": multiple_lines, "online_security": online_security,
        "online_backup": online_backup, "device_protection": device_protection,
        "tech_support": tech_support, "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies, "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges, "total_charges": total_charges,
        "internet_service": internet_service, "contract": contract,
        "payment_method": payment_method,
    }

    result = predict(inputs, threshold=threshold)

    st.subheader("Result")
    if result["prediction"] == 1:
        st.error(f"⚠️ Likely to churn — probability: {result['probability']:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {result['probability']:.1%}")

    st.progress(min(result["probability"], 1.0))
    st.caption(f"Decision threshold used: {threshold:.2f}")
