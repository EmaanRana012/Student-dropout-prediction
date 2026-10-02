import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="centered"
)

# ----------------------------
# Load model, scaler, columns
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("dropout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ----------------------------
# Custom styling
# ----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .risk-low { background-color: #e6f4ea; border: 1px solid #34a853; }
    .risk-medium { background-color: #fff8e1; border: 1px solid #f9a825; }
    .risk-high { background-color: #fdecea; border: 1px solid #d93025; }
    .risk-label {
        font-size: 1.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Student Dropout Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a student\'s information to estimate dropout risk</div>', unsafe_allow_html=True)

# ----------------------------
# Input form
# ----------------------------
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age at enrollment", min_value=16, max_value=70, value=20)
        sem1_approved = st.slider("1st Semester Units Approved", 0, 20, 5)
        sem2_approved = st.slider("2nd Semester Units Approved", 0, 20, 5)
        sem2_grade = st.slider("2nd Semester Grade (0-20)", 0.0, 20.0, 12.0)

    with col2:
        tuition_status = st.radio("Tuition fees up to date?", ["Yes", "No"])
        debtor_status = st.radio("Is the student a debtor?", ["No", "Yes"])
        scholarship_status = st.radio("Scholarship holder?", ["No", "Yes"])
        gender = st.radio("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

# ----------------------------
# Prediction logic
# ----------------------------
def risk_category(prob):
    if prob < 0.30:
        return "Low Risk", "risk-low", "#34a853"
    elif prob < 0.60:
        return "Medium Risk", "risk-medium", "#f9a825"
    else:
        return "High Risk", "risk-high", "#d93025"

def build_input_row():
    # Start from a neutral baseline (zeros), then override known fields.
    # Any features not explicitly collected from the user keep a default value of 0,
    # which is a simplification — for a production app you'd want a fuller form
    # or sensible dataset-driven defaults for every column.
    row = {col: 0 for col in feature_columns}

    row["Age at enrollment"] = age
    row["Curricular units 1st sem (approved)"] = sem1_approved
    row["Curricular units 2nd sem (approved)"] = sem2_approved
    row["Curricular units 2nd sem (grade)"] = sem2_grade
    row["Tuition fees up to date"] = 1 if tuition_status == "Yes" else 0
    row["Debtor"] = 1 if debtor_status == "Yes" else 0
    row["Scholarship holder"] = 1 if scholarship_status == "Yes" else 0
    row["Gender"] = 1 if gender == "Male" else 0

    return pd.DataFrame([row])[feature_columns]

if submitted:
    input_df = build_input_row()
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    pred_class = model.predict(input_scaled)[0]
    label, css_class, color = risk_category(prob)

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="risk-label" style="color:{color};">{label}</div>
        <div>Dropout Probability: <b>{prob*100:.1f}%</b></div>
        <div>Predicted Class: <b>{"Dropout" if pred_class == 1 else "Not Dropout"}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#e6f4ea"},
                {'range': [30, 60], 'color': "#fff8e1"},
                {'range': [60, 100], 'color': "#fdecea"},
            ],
        },
        title={'text': "Dropout Probability"}
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with a Logistic Regression model trained on the UCI Student Dropout and Academic Success dataset.")
