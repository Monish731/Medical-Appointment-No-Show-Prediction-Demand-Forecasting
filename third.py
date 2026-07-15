import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Medical Appointment Analytics",
    page_icon="🏥",
    layout="wide"
)

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    no_show_model = joblib.load("no_show_model.pkl")
    demand_model = joblib.load("demand_forecasting_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return no_show_model, demand_model, scaler


no_show_model, demand_model, scaler = load_models()

# ===============================
# Title
# ===============================
st.title("🏥 Medical Appointment No-Show Prediction & Demand Forecasting")
st.markdown("---")

tab1, tab2 = st.tabs(["🩺 No-Show Prediction", "📈 Demand Forecasting"])

# ==========================================================
# TAB 1
# ==========================================================
with tab1:
    st.header("Patient Appointment Prediction")

    col1, col2 = st.columns(2)

    with col1:
        specialty = st.selectbox("Specialty (Encoded)", list(range(8)))
        appointment_time = st.number_input("Appointment Time", value=9)
        gender = st.selectbox("Gender", [0, 1, 2])
        disability = st.number_input("Disability", value=0)
        place = st.number_input("Place ID", value=9917)
        appointment_shift = st.selectbox("Appointment Shift", [0, 1])
        age = st.slider("Age", 0, 100, 30)
        under_12 = 1 if age < 12 else 0
        over_60 = 1 if age >= 60 else 0
        companion = st.selectbox("Needs Companion", [0, 1])

    with col2:
        avg_temp = st.number_input("Average Temperature", value=25.0)
        avg_rain = st.number_input("Average Rain", value=0.0)
        max_temp = st.number_input("Maximum Temperature", value=30.0)
        max_rain = st.number_input("Maximum Rain", value=0.0)
        rainy_before = st.selectbox("Rainy Day Before", [0, 1])
        storm_before = st.selectbox("Storm Day Before", [0, 1])
        rain_intensity = st.selectbox("Rain Intensity", [0, 1, 2, 3])
        heat_intensity = st.selectbox("Heat Intensity", [0, 1, 2, 3, 4])
        hypertension = st.selectbox("Hypertension", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        alcoholism = st.selectbox("Alcoholism", [0, 1])
        handicap = st.selectbox("Handicap", [0, 1])
        scholarship = st.selectbox("Scholarship", [0, 1])
        sms = st.selectbox("SMS Received", [0, 1])

    st.markdown("---")
    predict_btn = st.button("Predict No-Show", use_container_width=True)

    if predict_btn:
        # ========================================
        # Feature Engineering
        # ========================================
        year = 2025
        month = 1
        day = 1
        weekday = 0
        dayofweek = weekday
        weekofyear = 1
        quarter = 1
        isweekend = 1 if weekday >= 5 else 0

        disease_count = hypertension + diabetes + alcoholism + handicap
        temp_difference = max_temp - avg_temp
        rain_difference = max_rain - avg_rain

        # ========================================
        # Create Input DataFrame
        # EXACT SAME ORDER AS TRAINING
        # ========================================
        input_df = pd.DataFrame({
            "specialty": [specialty],
            "appointment_time": [appointment_time],
            "gender": [gender],
            "disability": [disability],
            "place": [place],
            "appointment_shift": [appointment_shift],
            "age": [age],
            "under_12_years_old": [under_12],
            "over_60_years_old": [over_60],
            "patient_needs_companion": [companion],
            "average_temp_day": [avg_temp],
            "average_rain_day": [avg_rain],
            "max_temp_day": [max_temp],
            "max_rain_day": [max_rain],
            "rainy_day_before": [rainy_before],
            "storm_day_before": [storm_before],
            "rain_intensity": [rain_intensity],
            "heat_intensity": [heat_intensity],
            "Hipertension": [hypertension],
            "Diabetes": [diabetes],
            "Alcoholism": [alcoholism],
            "Handcap": [handicap],
            "Scholarship": [scholarship],
            "SMS_received": [sms],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Weekday": [weekday],
            "Disease_Count": [disease_count],
            "Temp_Difference": [temp_difference],
            "Rain_Difference": [rain_difference],
            "DayOfWeek": [dayofweek],
            "WeekOfYear": [weekofyear],
            "Quarter": [quarter],
            "IsWeekend": [isweekend],
        })

        # ========================================
        # Scale
        # ========================================
        input_scaled = scaler.transform(input_df)

        # ========================================
        # Prediction
        # ========================================
        prediction = no_show_model.predict(input_scaled)[0]
        probability = no_show_model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("⚠ High Risk of No-Show")
            else:
                st.success("✅ Patient Likely to Attend")
        with col2:
            st.metric("No-Show Probability", f"{probability * 100:.2f}%")

        st.progress(float(probability))

        st.markdown("---")
        st.subheader("Input Summary")
        st.dataframe(input_df)

# ==========================================================
# TAB 2
# ==========================================================
with tab2:
    st.header("Daily Appointment Demand Forecast")
    st.markdown("Predict the expected number of appointments for a selected date.")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=2020, max_value=2035, value=2025)
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.selectbox("Day", list(range(1, 32)))

    with col2:
        weekday = st.selectbox("Weekday", list(range(7)), help="0=Monday ... 6=Sunday")
        weekofyear = st.slider("Week Of Year", 1, 53, 1)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        isweekend = st.selectbox("Weekend", [0, 1])

    forecast_btn = st.button("Forecast Demand", use_container_width=True)

    if forecast_btn:
        demand_df = pd.DataFrame({
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Weekday": [weekday],
            "WeekOfYear": [weekofyear],
            "Quarter": [quarter],
            "IsWeekend": [isweekend],
        })

        prediction = demand_model.predict(demand_df)[0]

        st.success("Forecast Generated Successfully")
        st.metric("Expected Appointments", f"{prediction:.0f}")

        st.subheader("Forecast Details")
        st.dataframe(demand_df)

        chart = pd.DataFrame({
            "Category": ["Predicted Demand"],
            "Appointments": [prediction],
        })
        st.bar_chart(chart.set_index("Category"))

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.markdown(
    """
### 📊 Project Information

**Project Title**
Medical Appointment No-Show Prediction & Demand Forecasting

**Models Used**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Random Forest Regressor

**Developed Using**
- Python
- Scikit-Learn
- XGBoost
- Streamlit
- Pandas
- NumPy
"""
)