import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os


st.set_page_config(
    page_title="CardioGuard | Heart Health AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Loads data. If file missing, generates synthetic data for demo purposes.
    """
    try:
        df = pd.read_csv("cleaned_cardio.csv")
    except FileNotFoundError:
        st.warning("⚠️ 'cleaned_cardio.csv' not found. Generating synthetic data for demonstration.")
    
        np.random.seed(42)
        rows = 1000
        data = {
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
        }
        df = pd.DataFrame(data)

    if df['age'].mean() > 150: 
        df['age_years'] = (df['age'] / 365.25).round(1)
    else:
        df['age_years'] = df['age']

    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    return df

@st.cache_resource
def train_model(df):
    """
    Trains the model and caches it so it doesn't re-train on every reload.
    """
    feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    
    X = df[feature_cols]
    y = df['cardio']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, acc, y_test, y_pred


df = load_data()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
st.sidebar.title("CardioGuard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Dataset", "📊 Data Analysis", "🧠 Model Studio", "🩺 Health Risk Test"]
)
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")

# HOME PAGE 
if page == "🏠 Home":
    st.title("🫀 Cardiovascular Disease Risk Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to CardioGuard
        Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection and lifestyle changes are crucial for prevention.
        
        **This intelligent system allows you to:**
        * **Analyze** historical patient data to find trends.
        * **Visualize** risk factors like Blood Pressure, BMI, and Cholesterol.
        * **Predict** your own risk using machine learning algorithms.
        """)
    
        st.image("assets\Heart Anatomy.jpg", 
                 caption="Heart Anatomy", 
                 use_container_width=True)
        
        st.info("Navigate using the sidebar to explore the dataset or take the health test.")

    with col2:
        st.markdown("### 📈 Quick Stats")
        st.metric(label="Total Patients Records", value=f"{df.shape[0]:,}")
        st.metric(label="High Risk Cases", value=f"{df['cardio'].sum():,}")
        st.metric(label="Average Patient Age", value=f"{int(df['age_years'].mean())} years")

# DATASET PAGE
elif page == "📂 Dataset":
    st.title("📂 Dataset Overview")
    st.markdown("View the raw data used to train the machine learning models.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isna().sum().sum())

    st.divider()
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 View Statistical Summary"):
        st.write(df.describe())

# DATA ANALYSIS PAGE 
elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    with tab1:
        st.subheader("Distribution of Features")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="age_years", nbins=20, title="Age Distribution", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='cardio', title="Target Distribution (0=Healthy, 1=Risk)", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Features vs. Disease Status")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="cardio", y="age_years", color="cardio", title="Age vs Cardiovascular Disease")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x="cardio", y="weight", color="cardio", title="Weight vs Cardiovascular Disease")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.subheader("Complex Relationships")
        st.markdown("### Blood Pressure Analysis")
        fig = px.scatter(df, x="ap_hi", y="ap_lo", color="cardio", 
                         title="Systolic vs Diastolic BP (Colored by Risk)",
                         labels={'ap_hi': 'Systolic BP', 'ap_lo': 'Diastolic BP'},
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# MODEL TRAINING PAGE 
elif page == "🧠 Model Studio":
    st.title("🧠 Model Training & Evaluation")
    
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training Random Forest Classifier..."):
            model, scaler, acc, y_test, y_pred = train_model(df)
            
        st.success(f"Model Trained Successfully! Accuracy: {acc:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['Healthy', 'Disease'], y=['Healthy', 'Disease'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Feature Importance")
            feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=True)
            
            fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top Risk Factors")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("💾 Save Model Weights"):
            joblib.dump(model, "cardio_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            st.success("Model saved to disk!")
    else:
        st.info("Click the button above to train the model.")

# PREDICTION PAGE 
elif page == "🩺 Health Risk Test":
    st.title("🩺 Health Risk Assessment")
    st.markdown("Enter your biological data below to get a risk result.")
    
    st.image("assets\Cardiovascular Medical Checkup.jpg", 
             width=400, caption="Medical Checkup")

    # Check if model exists
    if not os.path.exists("cardio_model.pkl"):
        st.warning("⚠️ No model found! Please go to the 'Model Studio' page and train the model first.")
    else:
        model = joblib.load("cardio_model.pkl")
        scaler = joblib.load("scaler.pkl")

        with st.form("risk_form"):
            st.subheader("1. Personal Information")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 18, 100, 30)
            gender = c2.selectbox("Gender", ["Female", "Male"])
            height = c3.number_input("Height (cm)", 100, 250, 170)
            
            st.subheader("2. Vitals & Biometrics")
            c4, c5, c6 = st.columns(3)
            weight = c4.number_input("Weight (kg)", 30, 200, 70)
            ap_hi = c5.number_input("Systolic BP (High)", 80, 250, 120, help="Normal is ~120")
            ap_lo = c6.number_input("Diastolic BP (Low)", 40, 150, 80, help="Normal is ~80")
            
            st.subheader("3. Lifestyle & Lab Results")
            c7, c8 = st.columns(2)
            cholesterol = c7.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            gluc = c8.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
            
            c9, c10, c11 = st.columns(3)
            smoke = c9.checkbox("Do you smoke?")
            alco = c10.checkbox("Alcohol intake?")
            active = c11.checkbox("Physical Activity?")

            submitted = st.form_submit_button("🔍 Analyze Risk")

        if submitted:
            # Data Preprocessing Logic
            gender_val = 2 if gender == "Male" else 1
            chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            bmi_val = weight / ((height / 100) ** 2)
            
            user_input = [[
                age, gender_val, height, weight, ap_hi, ap_lo, 
                chol_map[cholesterol], gluc_map[gluc], 
                int(smoke), int(alco), int(active), bmi_val
            ]]
            
            user_scaled = scaler.transform(user_input)
            prediction = model.predict(user_scaled)
            probability = model.predict_proba(user_scaled)[0][1]

            st.divider()
            st.subheader("Your Results")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediction[0] == 1:
                    st.error("High Risk Detected")
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
                else:
                    st.success("Low Risk Detected")
                    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966343.png", width=100)

            with col_res2:
                st.write(f"**Risk Probability Score:** {probability:.1%}")
                st.progress(probability)
                
                st.markdown(f"""
                **Health Summary:**
                - **BMI:** {bmi_val:.1f} ($kg/m^2$)
                - **Blood Pressure:** {ap_hi}/{ap_lo}
                """)
                
                if probability > 0.5:
                    st.warning("⚠️ Your indicators suggest a higher risk. Please consult a cardiologist.")
                else:
                    st.success("✅ Keep up the healthy lifestyle!")