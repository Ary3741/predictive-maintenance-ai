# 🏭 Predictive Maintenance with Explainable AI (XAI)

> An end-to-end Machine Learning system that predicts industrial machine failure from sensor data — and explains **why** using SHAP (SHapley Additive exPlanations).

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🔗 Live Demo

👉 **[Launch App](https://YOUR_STREAMLIT_URL_HERE)**

---

## 📌 Problem Statement

In industrial environments, unplanned machine failures cost companies millions in downtime, repairs, and lost productivity. Traditional approaches are reactive — machines are fixed after they break, or replaced on fixed schedules regardless of their actual condition.

This project addresses that gap by building a **predictive maintenance system** that:
- Monitors real-time sensor readings (temperature, torque, speed, tool wear)
- Predicts machine failure **before it happens**
- Explains **which sensor readings** are driving the risk — giving engineers actionable insight, not just a black-box alert

---

## 🎯 Key Features

- ✅ **Real-time failure prediction** from 5 sensor inputs
- ✅ **SHAP Explainability** — waterfall plots showing exactly why the model flagged a machine
- ✅ **Physics-informed feature engineering** — Power, Temperature Difference, Wear-Torque interaction features
- ✅ **Plain English explanation** — translates ML output into actionable maintenance recommendations
- ✅ **Color-coded risk levels** — Green / Yellow / Red risk indicators
- ✅ **Interactive Streamlit dashboard** — no coding required to use

---

## 🖥️ App Preview

```
┌─────────────────────────────────────────────────────┐
│  🏭 Predictive Maintenance Dashboard                │
│                                                     │
│  ⚙️ Sidebar: Enter Sensor Readings                 │
│  ├── Air Temperature (K)          [slider]          │
│  ├── Process Temperature (K)      [slider]          │
│  ├── Rotational Speed (rpm)       [slider]          │
│  ├── Torque (Nm)                  [slider]          │
│  └── Tool Wear (min)              [slider]          │
│                                                     │
│  🔍 Prediction Result                              │
│  ├── Failure Probability: 87.3%                    │
│  ├── ⚠️ HIGH FAILURE RISK                          │
│  └── 🔴 Risk Bar                                   │
│                                                     │
│  🧠 SHAP Explanation (Waterfall Plot)              │
│  💬 Plain English Recommendation                   │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- 10,000 rows of synthetic industrial sensor data
- 5 raw sensor features + failure labels
- Class distribution: ~96.6% No Failure / ~3.4% Failure (imbalanced)

### Features Used

| Feature | Type | Description |
|---|---|---|
| Type | Categorical | Machine quality variant (L/M/H) |
| Air Temperature (K) | Sensor | Ambient air temperature |
| Process Temperature (K) | Sensor | Machine process temperature |
| Rotational Speed (rpm) | Sensor | Spindle rotational speed |
| Torque (Nm) | Sensor | Rotational force applied |
| Tool Wear (min) | Sensor | Cumulative tool usage time |
| **Power** | Engineered | Torque × Rotational Speed |
| **Temp Difference** | Engineered | Process Temp − Air Temp |
| **Wear Torque** | Engineered | Tool Wear × Torque |
| **Wear Power** | Engineered | Tool Wear × Power |

---

## 🧠 Model Performance

| Metric | Random Forest |
|---|---|
| Accuracy | 99.0% |
| ROC-AUC | 96.6% |
| Recall (Failure class) | 81% |
| Precision (Failure class) | 89% |
| Failures caught (out of 68) | 55 |

> **Why Recall matters most:** Missing a real failure is far more costly than a false alarm. 81% recall means we prevent 55 out of every 68 machine breakdowns before they happen.

---

## 🔍 Explainability — SHAP Results

### Global Feature Importance
Top predictors identified by SHAP across the entire dataset:

1. **Power** — most influential engineered feature
2. **Rotational Speed** — low RPM under load = high risk
3. **Torque** — high torque = mechanical strain
4. **Temp Difference** — shrinking gap = poor heat dissipation
5. **Wear Torque** — compound stress indicator

### Local Explanation (Sample Prediction)
For a flagged machine, SHAP explained:
- Power: **+0.29** → pushing failure probability UP
- Torque: **+0.26** → pushing failure probability UP
- Final prediction: **100% failure confidence**

This transforms a black-box model into an **actionable diagnostic tool**.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Boosting | XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Ary3741/predictive-maintenance-xai.git
cd predictive-maintenance-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
predictive-maintenance-xai/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/
│   ├── rf_model.pkl                # Trained Random Forest model
│   └── feature_names.pkl           # Feature names for SHAP
│
├── data/
│   └── ai4i_engineered.csv         # Cleaned + engineered dataset
│
└── notebooks/
    ├── step1_exploration.py        # Data loading & inspection
    ├── step2_eda.py                # Exploratory data analysis
    ├── step3_features.py           # Feature engineering
    ├── step4_model.py              # Model training & evaluation
    └── step5_shap.py               # SHAP explainability
```

---

## 📈 Business Impact

Every unplanned machine breakdown in a typical factory costs between **₹2–10 lakh** in downtime and emergency repairs. This system:

- Catches **81% of failures** before they occur
- Provides **specific sensor-level reasoning** so engineers know exactly what to inspect
- Reduces unnecessary preventive maintenance by flagging only at-risk machines
- Can be integrated directly with **IoT sensor pipelines** for real-time automated monitoring

---

## 🔮 Future Scope

- [ ] Real-time IoT sensor data integration (MQTT / REST API)
- [ ] Multi-class failure type prediction (TWF, HDF, PWF, OSF)
- [ ] Time-series anomaly detection on rolling sensor windows
- [ ] Mobile-friendly dashboard
- [ ] Email/SMS alert system for flagged machines
- [ ] Docker containerization for enterprise deployment

---

## 👨‍💻 Author

**Aryan**
Computer Science Engineering Undergraduate | BVCOE, New Delhi

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is intended for educational and decision-support purposes only. Always consult qualified engineers before making maintenance decisions based on model predictions.

---

<p align="center">
  Built with ❤️ using Python, Scikit-learn, SHAP, and Streamlit
</p>
