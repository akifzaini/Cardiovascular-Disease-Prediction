import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

df = pd.read_csv("cleaned_cardio.csv")

st.sidebar.title("🫀 CVD Prediction App")
st.sidebar.markdown("Navigate through the application")

page = st.sidebar.radio(
    "Menu",
    ["🏠 Home", "📂 Dataset", "📊 Data Analysis", "🧠 Model Training", "🩺 Health Risk Test"]
)

#HOME PAGE

if page == "🏠 Home":
    st.title("🫀 Cardiovascular Disease Risk Prediction")

    st.markdown("""
    ### Welcome!
    This web application uses **machine learning** to predict the risk of
    cardiovascular disease based on personal health information.

    #### What can you do here?
    - 📂 Explore the dataset
    - 📊 Analyze health trends
    - 🧠 Train prediction models
    - 🩺 Test cardiovascular disease risk
    """)

#DATASET PAGE
elif page == "📂 Dataset":
    st.title("📂 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    with st.expander("📊 Statistical Summary"):
        st.write(df.describe())


#DATA ANALYSIS PAGE
elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        df["cardio"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Cardiovascular Disease")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Age vs Disease")
        fig, ax = plt.subplots()
        df.boxplot(column="age", by="cardio", ax=ax)
        st.pyplot(fig)

    st.subheader("Blood Pressure Relationship")
    fig, ax = plt.subplots()
    ax.scatter(df["ap_hi"], df["ap_lo"], c=df["cardio"])
    ax.set_xlabel("Systolic BP")
    ax.set_ylabel("Diastolic BP")
    st.pyplot(fig)

#MODEL TRAINING PAGE
elif page == "🧠 Model Training":
    st.title("🧠 Model Training & Evaluation")

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model Accuracy: {acc:.2f}")

    with st.expander("📉 Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    if st.button("💾 Save Model for Prediction"):
        joblib.dump(model, "cardio_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model saved successfully!")

#PREDICTION PAGE 
elif page == "🩺 Health Risk Test":
    st.title("🩺 Cardiovascular Health Risk Test")
    st.info("Please enter your health information below")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 1, 120)
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender = 0 if gender == "Female" else 1

        height = st.number_input("Height (cm)", 100, 250)
        weight = st.number_input("Weight (kg)", 30, 200)

        ap_hi = st.number_input("Systolic Blood Pressure", 80, 250)
        ap_lo = st.number_input("Diastolic Blood Pressure", 40, 200)

    with col2:
        cholesterol = st.selectbox(
            "Cholesterol Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        gluc = st.selectbox(
            "Glucose Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        smoke = st.selectbox("Smoking", ["No", "Yes"])
        smoke = 0 if smoke == "No" else 1

        alco = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        alco = 0 if alco == "No" else 1

        active = st.selectbox("Physically Active", ["No", "Yes"])
        active = 0 if active == "No" else 1

    #BMI CALCULATION
    bmi = weight / ((height / 100) ** 2)

    user_data = [[
        age, gender, height, weight,
        ap_hi, ap_lo,
        cholesterol, gluc,
        smoke, alco, active,
        bmi
    ]]

    if st.button("🔍 Check My Risk"):
        model = joblib.load("cardio_model.pkl")
        scaler = joblib.load("scaler.pkl")

        user_scaled = scaler.transform(user_data)
        prediction = model.predict(user_scaled)

        st.divider()

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Cardiovascular Disease")
            st.markdown("👉 Consider consulting a healthcare professional.")
        else:
            st.success("✅ Low Risk of Cardiovascular Disease")
            st.markdown("👍 Keep maintaining a healthy lifestyle!")
