import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

st.set_page_config(
    page_title="CardioGuard | Heart Health AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


#DATA LOADING 
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_cardio.csv")
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Using synthetic data.")
        np.random.seed(42)
        rows = 1000
        df = pd.DataFrame({
            'age': np.random.randint(10000, 23000, rows),
            'gender': np.random.randint(1, 3, rows),
            'height': np.random.randint(150, 190, rows),
            'weight': np.random.randint(50, 120, rows),
            'ap_hi': np.random.randint(90, 180, rows),
            'ap_lo': np.random.randint(60, 110, rows),
            'cholesterol': np.random.randint(1, 4, rows),
            'gluc': np.random.randint(1, 4, rows),
            'smoke': np.random.randint(0, 2, rows),
            'alco': np.random.randint(0, 2, rows),
            'active': np.random.randint(0, 2, rows),
            'cardio': np.random.randint(0, 2, rows)
        })

    df['age_years'] = (df['age'] / 365.25).round(1) if df['age'].mean() > 150 else df['age']

    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['mean_arterial_pressure'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )

    return df


df = load_data()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
st.sidebar.title("CardioGuard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Dataset", "📊 Data Analysis", "🧠 Model Studio", "🩺 Health Risk Test"]
)

st.sidebar.markdown("---")
st.sidebar.info("Educational use only. Not a medical diagnosis.")

if page == "🏠 Home":
    st.title("🫀 Cardiovascular Disease Risk Prediction")
    st.markdown("""
    **CardioGuard** analyzes cardiovascular risk factors using machine learning  
    and exploratory data analysis to support early disease detection.
    """)


#DATASET
elif page == "📂 Dataset":
    st.title("📂 Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 Statistical Summary"):
        st.write(df.describe())


#DATA ANALYSIS 
elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='cardio', data=df, ax=ax)
        ax.set_title("Cardiovascular Disease Distribution")
        ax.set_xlabel("Cardio (0 = No, 1 = Yes)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x='cardio', y='pulse_pressure', data=df, ax=ax)
        ax.set_title("Pulse Pressure vs Cardiovascular Disease")
        ax.set_ylabel("Pulse Pressure (mmHg)")
        st.pyplot(fig)

    st.subheader("BMI Distribution by Disease Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x='cardio', y='bmi', data=df, inner="quartile", ax=ax)
    ax.set_xlabel("Cardio (0 = No, 1 = Yes)")
    ax.set_ylabel("BMI")
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    corr_features = [
        'age_years', 'bmi', 'ap_hi', 'ap_lo',
        'pulse_pressure', 'mean_arterial_pressure',
        'cholesterol', 'gluc', 'active', 'cardio'
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[corr_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )
    st.pyplot(fig)


# MODEL STUDIO 
elif page == "🧠 Model Studio":
    st.title("🧠 Model Studio")

    X = df.drop(['cardio', 'bmi_category'], axis=1)
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    if st.button("🚀 Train Model"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=500)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "cardio_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        st.subheader("🧠 Model Insight")
        st.markdown("""
        - Blood pressure indicators strongly influence predictions  
        - BMI and metabolic features increase cardiovascular risk  
        - Lifestyle variables contribute indirectly
        """)


#PREDICTION
elif page == "🩺 Health Risk Test":
    st.title("🩺 Health Risk Assessment")

    if not os.path.exists("cardio_model.pkl"):
        st.warning("⚠️ Please train a model first.")
    else:
        model = joblib.load("cardio_model.pkl")
        scaler = joblib.load("scaler.pkl")

        with st.form("risk_form"):
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", ["Female", "Male"])
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            ap_hi = st.number_input("Systolic BP", 80, 250, 120)
            ap_lo = st.number_input("Diastolic BP", 40, 150, 80)

            submitted = st.form_submit_button("Analyze Risk")

        if submitted:
            bmi = weight / ((height / 100) ** 2)
            gender_val = 2 if gender == "Male" else 1

            user_input = [[
                age, gender_val, height, weight,
                ap_hi, ap_lo, 1, 1, 0, 0, 1,
                bmi, ap_hi - ap_lo, (ap_hi + 2 * ap_lo) / 3
            ]]

            user_scaled = scaler.transform(user_input)
            prediction = model.predict(user_scaled)[0]
            probability = model.predict_proba(user_scaled)[0][1]

            if prediction == 1:
                st.error(f"High Risk Detected ({probability:.1%})")
            else:
                st.success(f"Low Risk Detected ({probability:.1%})")
