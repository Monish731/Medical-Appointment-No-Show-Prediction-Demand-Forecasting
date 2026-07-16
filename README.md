# 🏥 Medical Appointment No-Show Prediction & Demand Forecasting

## 📌 Project Overview

This project uses Machine Learning to solve two important healthcare problems:

1. **Patient No-Show Prediction**
   - Predicts whether a patient is likely to attend or miss a scheduled appointment.

2. **Appointment Demand Forecasting**
   - Predicts the expected number of appointments for a future date to help hospitals manage resources efficiently.

The project is deployed using **Streamlit**, allowing hospital staff to interact with the trained machine learning models through a simple web interface.

---

# 🎯 Objectives

- Predict patient no-shows before the appointment.
- Forecast future appointment demand.
- Reduce hospital operational costs.
- Improve doctor and staff scheduling.
- Increase resource utilization.
- Improve patient satisfaction.

---

# 📂 Project Structure

```
Medical_Appointment_Project/
│
├── app.py
├── Medical_appointment_data.csv
├── no_show_model.pkl
├── demand_forecasting_model.pkl
├── scaler.pkl
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── No_Show_Prediction.ipynb
│   └── Demand_Forecasting.ipynb
│
└── images/
```

---

# 📊 Dataset

The dataset contains historical hospital appointment information including:

- Patient Details
- Appointment Details
- Weather Information
- Medical Conditions
- Date Information
- Appointment Status

Target Variable:

```
no_show

0 → Patient Attended

1 → Patient Missed Appointment
```

---

# 🔍 Features Used

### Patient Information

- Specialty
- Gender
- Age
- Disability
- Place
- Appointment Shift
- Appointment Time

### Weather Features

- Average Temperature
- Maximum Temperature
- Average Rain
- Maximum Rain
- Rain Intensity
- Heat Intensity
- Rainy Day Before
- Storm Day Before

### Medical Features

- Hypertension
- Diabetes
- Alcoholism
- Handicap
- Scholarship
- SMS Received

### Engineered Features

- Disease Count
- Temperature Difference
- Rain Difference
- Under 12 Years Old
- Over 60 Years Old
- Weekday
- Week of Year
- Quarter
- Weekend

---

# 🤖 Machine Learning Models

## No-Show Prediction

Classification Models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost ✅ Best Model

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Demand Forecasting

Regression Model:

- Random Forest Regressor

Evaluation Metrics:

- MAE
- RMSE
- R² Score

---

# 📈 Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

Top Important Features:

- Place
- Specialty
- Age
- Appointment Time
- Weather
- Rain
- Temperature

---

# 🌐 Streamlit Web Application

The application contains two modules.

## 🩺 No-Show Prediction

Input:

- Patient Details
- Medical History
- Weather Information

Output:

- Attend / No-Show Prediction
- No-Show Probability

---

## 📈 Demand Forecasting

Input:

- Year
- Month
- Day
- Weekday

Output:

- Expected Number of Appointments

---

# 📷 Sample Output

### No-Show Prediction

```
Prediction

⚠ High Risk of No-Show

Probability

57.79%
```

### Demand Forecasting

```
Expected Appointments

142 Patients
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Medical-Appointment-Analytics.git
```

Move into project folder

```bash
cd Medical-Appointment-Analytics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📊 Business Benefits

This project helps hospitals to:

- Reduce appointment no-shows.
- Improve doctor scheduling.
- Forecast patient demand.
- Optimize hospital resources.
- Reduce waiting time.
- Improve patient satisfaction.
- Lower operational costs.

---

# 🔮 Future Scope

- Real-time Weather API
- SMS Reminder Integration
- Cloud Deployment
- Hospital Management System Integration
- Deep Learning Models
- Mobile Application

---

# 👨‍💻 Author

**Monish Sai**

Machine Learning Project

Medical Appointment No-Show Prediction & Demand Forecasting
