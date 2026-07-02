import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#OPENING PAGE STREAMLIT 
st.set_page_config(
    page_title="Machine Failure Predictor",
    layout="wide"
)

#LOAD ALL THE MODELS 
@st.cache_resource
def load_model():
    with open("C:\Users\Aryan\OneDrive\Desktop\predictive maintenance\models\rf_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("C:\Users\Aryan\OneDrive\Desktop\predictive maintenance\models\feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()
explainer = shap.TreeExplainer(model)

#PAGE HEADER 
st.title("Predictive Maintenance Dashboard")
st.markdown("###Machine Failure Prediction with AI")
st.markdown("""
This tool uses a **Random Forest model** trained on industrial sensor data
to predict whether a machine is at risk of failure — and explains **why**
using SHAP (SHapley Additive exPlanations).
""")
st.divider()

#PAGE SIDEBAR INPUT READINGS 
st.sidebar.title("Enter Sensor Readings")
st.sidebar.markdown("Adjust the sliders to match current machine readings:")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    options=[0, 1, 2],
    format_func=lambda x: {0: "L — Low Quality", 1: "M — Medium Quality", 2: "H — High Quality"}[x]
)

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    min_value=295.0, max_value=305.0, value=300.0, step=0.1
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    min_value=305.0, max_value=315.0, value=310.0, step=0.1
)

rot_speed = st.sidebar.slider(
    "Rotational Speed (rpm)",
    min_value=1168, max_value=2886, value=1500, step=1
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=3.8, max_value=76.6, value=40.0, step=0.1
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    min_value=0, max_value=253, value=100, step=1
)

#IN-BUILT CALCULATIONS 
temp_difference = process_temp - air_temp
power = torque * rot_speed
wear_torque = tool_wear * torque
wear_power = tool_wear * power

# Build input dataframe
input_data = pd.DataFrame([[
    machine_type, air_temp, process_temp,
    rot_speed, torque, tool_wear,
    temp_difference, power, wear_torque, wear_power
]], columns=feature_names)

#RESULT ANALYSIS
st.markdown("## Prediction Result")

col1, col2, col3 = st.columns(3)

prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

with col1:
    st.metric(
        label="Failure Probability",
        value=f"{prob * 100:.1f}%"
    )

with col2:
    if prediction == 1:
        st.error("⚠️ HIGH FAILURE RISK")
    else:
        st.success("✅ MACHINE HEALTHY")

with col3:
    st.metric(
        label="Confidence",
        value=f"{max(prob, 1-prob) * 100:.1f}%"
    )

# Risk level bar
st.markdown("### Risk Level")
if prob < 0.3:
    st.progress(prob, text=f"🟢 Low Risk ({prob*100:.1f}%)")
elif prob < 0.6:
    st.progress(prob, text=f"🟡 Medium Risk ({prob*100:.1f}%)")
else:
    st.progress(prob, text=f"🔴 High Risk ({prob*100:.1f}%)")

st.divider()

#DISPLAY SENSOR READINGS
st.markdown("## Current Sensor Readings")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Air Temp (K)", f"{air_temp:.1f}")
col2.metric("Process Temp (K)", f"{process_temp:.1f}")
col3.metric("Speed (rpm)", f"{rot_speed}")
col4.metric("Torque (Nm)", f"{torque:.1f}")
col5.metric("Tool Wear (min)", f"{tool_wear}")

st.divider()

#SHAP EXPLAINATION (RANDOM FOREST MODEL)
st.markdown("## Why Did the Model Predict This?")
st.markdown("SHAP values show which sensor readings drove this prediction:")

shap_values = explainer.shap_values(input_data)
shap_vals_failure = shap_values[:, :, 1]

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value[1],
    shap_vals_failure[0],
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Explanation — What drove this prediction?")
plt.tight_layout()
st.pyplot(fig)
plt.close()

#ENGLISH EXPLAINATION 
st.divider()
st.markdown("## CONCLLUSION DERIVED")

shap_series = pd.Series(shap_vals_failure[0], index=feature_names)
top_positive = shap_series[shap_series > 0].sort_values(ascending=False)
top_negative = shap_series[shap_series < 0].sort_values(ascending=True)

if prediction == 1:
    st.error(f"""
    ⚠️ **This machine is at HIGH RISK of failure.**

    The main reasons are:
    - **{top_positive.index[0] if len(top_positive) > 0 else 'N/A'}** is critically high
      → pushing failure probability UP by {top_positive.iloc[0]:.3f}
    - **{top_positive.index[1] if len(top_positive) > 1 else 'N/A'}** is also elevated
      → pushing failure probability UP by {top_positive.iloc[1]:.3f}

    🔧 **Recommended Action:** Schedule immediate maintenance inspection.
    """)
else:
    st.success(f"""
    ✅ **This machine appears healthy.**

    The main stabilizing factors are:
    - **{top_negative.index[0] if len(top_negative) > 0 else 'N/A'}** is within safe range
      → keeping failure probability LOW by {abs(top_negative.iloc[0]):.3f}

    📋 **Recommended Action:** Continue normal monitoring schedule.
    """)

#PAGE FOOTER 
st.divider()
st.markdown("""
<small>
⚠️ This tool is for educational and decision-support purposes only.
Always consult qualified engineers before making maintenance decisions.
Built with Random Forest + SHAP Explainability.
</small>
""", unsafe_allow_html=True)
