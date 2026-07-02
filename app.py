import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Intervention System", page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout Early Intervention System")
st.write("Predict a student's risk of dropping out based on early academic performance and financial status.")

# ----------------------------
# Load model (stop app gracefully if missing)
# ----------------------------
model = None
try:
    model = joblib.load('dropout_model.pkl')
except FileNotFoundError:
    st.error("⚠️ Model file 'dropout_model.pkl' not found. Please ensure it is in the same folder as app.py")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Enter Student Data")
st.sidebar.write("*(Top Predictive Features)*")

tuition = st.sidebar.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
sem1_approved = st.sidebar.slider("1st Semester Units Approved", 0, 10, 5)
sem2_approved = st.sidebar.slider("2nd Semester Units Approved", 0, 10, 5)

total_units = sem1_approved + sem2_approved
tuition_num = 1 if tuition == "Yes" else 0

analyze_clicked = st.sidebar.button("Analyze Student Risk")

# ----------------------------
# Main logic — only runs after button click
# ----------------------------
if analyze_clicked:
    # Build the feature vector for the model (34 features expected)
    input_data = np.zeros(34)
    input_data[0] = tuition_num
    input_data[1] = sem1_approved
    input_data[2] = sem2_approved

    dropout_risk = None
    try:
        input_df = pd.DataFrame([input_data])
        probabilities = model.predict_proba(input_df)[0]
        dropout_risk = probabilities[1] * 100
    except Exception as e:
        st.error("Model unavailable. Please check the inputs or contact support.")
        dropout_risk = None

    if dropout_risk is not None:
        st.subheader("📊 Analysis Results")

        if dropout_risk >= 75:
            st.error(f"High Risk of Dropout: {dropout_risk:.1f}%")
            st.markdown(
                """
            **🚨 Recommended Interventions:**
            * Schedule immediate mandatory academic counseling.
            * Review tuition payment plans and financial aid options.
            * Alert academic advisor for 1-on-1 check-in.
            """
            )
        elif dropout_risk >= 40:
            st.warning(f"Medium Risk of Dropout: {dropout_risk:.1f}%")
            st.markdown(
                """
            **⚠️ Recommended Interventions:**
            * Assign a peer mentor for the upcoming semester.
            * Suggest enrollment in remedial study sessions for challenging subjects.
            """
            )
        else:
            st.success(f"Low Risk of Dropout: {dropout_risk:.1f}%")
            st.markdown(
                """
            **✅ Recommended Interventions:**
            * No immediate action needed. Student is on track.
            """
            )

        st.caption("Model Accuracy (validation set): 92%")
else:
    st.info("👈 Enter student data in the sidebar and click **Analyze Student Risk** to see results.")

# ----------------------------
# About / Responsible use section (single expander, de-duplicated)
# ----------------------------
with st.expander("About the Predictions & Responsible Use"):
    st.write("**Interpretability:** This tool provides risk scores based on historical data patterns "
             "to help identify students who may need additional academic assistance.")
    st.write("**Limitations:** This model is based on historical data and may not account for "
             "unforeseen personal or external factors affecting student performance.")
    st.write("**Fallback Responses:** If the system encounters an error during data processing, "
             "a default user-friendly message will be displayed instead of raw errors.")
    st.write("**Responsible Use:** Predictions should be used to inform support strategies, "
             "not as the sole basis for automated or punitive decisions.")
