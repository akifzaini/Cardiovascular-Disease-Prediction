import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Card-style containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ef4444;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #dc2626;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

#DATA LOADING 
@st.cache_data
def load_data():
    # Replace with your actual file path
    return pd.read_csv("cleaned_cardio.csv")

try:
    df = load_data()
except:
    st.error("Dataset not found. Please ensure 'cleaned_cardio.csv' is in the directory.")
    st.stop()

#SIDEBAR NAVIGATION 
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("CardioGuard AI")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "📂 Dataset", "📊 Analysis", "🧠 Training", "🩺 Risk Test"],
        index=0
    )
    st.markdown("---")
    st.info("💡 **Tip:** Regular checkups and a balanced diet significantly reduce CVD risk.")

#HOME PAGE
if page == "🏠 Home":
    st.title("❤️ Cardiovascular Health Analytics")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Predict. Prevent. Protect.
        This platform leverages **Logistic Regression Machine Learning** to assess cardiovascular risk factors. 
        By analyzing biometric data and lifestyle habits, CardioGuard provides instant insights into heart health.
        
        #### How it works:
        1. **Data Exploration:** View the statistical trends of thousands of patients.
        2. **Model Training:** Train the AI on the latest clinical data.
        3. **Risk Assessment:** Input your vitals to get an instant risk score.
        """)
        if st.button("Start Risk Test Now"):
            st.toast("Navigating to Risk Test...")
    
    with col2:
        st.image("https://img.freepik.com/free-vector/healthy-heart-concept-illustration_114360-10903.jpg")

#DATASET PAGE 
elif page == "📂 Dataset":
    st.title("📂 Clinical Data Overview")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Patients", f"{df.shape[0]:,}")
    m2.metric("Health Metrics", df.shape[1])
    m3.metric("Avg Age", f"{int(df['age'].mean())} yrs")

    tab1, tab2 = st.tabs(["🔍 Data Preview", "📈 Statistics"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.write(df.describe())

#ANALYSIS PAGE
elif page == "📊 Analysis":
    st.title("📊 Exploratory Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig_dist = px.pie(df, names='cardio', title="CVD Prevalence (0: Healthy, 1: At Risk)",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with c2:
        fig_box = px.box(df, x="cardio", y="age", title="Age Distribution by Heart Health",
                        color="cardio", color_discrete_sequence=["#22c55e", "#ef4444"])
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Blood Pressure Mapping")
    fig_scatter = px.scatter(df.sample(1000), x="ap_hi", y="ap_lo", color="cardio",
                            labels={"ap_hi": "Systolic", "ap_lo": "Diastolic"},
                            title="Systolic vs Diastolic BP (Sampled)",
                            color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig_scatter, use_container_width=True)

#TRAINING PAGE 
elif page == "🧠 Training":
    st.title("🧠 AI Model Management")
    
    with st.status("Preparing data and training model...", expanded=True) as status:
        X = df.drop("cardio", axis=1)
        y = df["cardio"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        status.update(label=f"Training Complete! Accuracy: {acc:.2%}", state="complete")

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"### Model Accuracy: {acc:.2%}")
        if st.button("💾 Deploy & Save Model"):
            joblib.dump(model, "cardio_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            st.balloons()
    
    with col2:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Healthy', 'Risk'], y=['Healthy', 'Risk'],
                          color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

#RISK TEST PAGE 
elif page == "🩺 Risk Test":
    st.title("🩺 Personalized Health Assessment")
    st.markdown("Enter your details below for a real-time risk evaluation.")

    with st.container():
        t1, t2 = st.tabs(["📏 Biometrics", "🚬 Lifestyle"])
        
        with t1:
            col1, col2 = st.columns(2)
            age = col1.slider("Age", 18, 100, 40)
            gender = col2.selectbox("Gender", ["Female", "Male"])
            height = col1.number_input("Height (cm)", 100, 250, 170)
            weight = col2.number_input("Weight (kg)", 30, 200, 70)
            ap_hi = col1.number_input("Systolic BP", 80, 250, 120)
            ap_lo = col2.number_input("Diastolic BP", 40, 200, 80)
            
        with t2:
            col1, col2 = st.columns(2)
            chol = col1.select_slider("Cholesterol", options=["Normal", "Above Normal", "High"])
            gluc = col2.select_slider("Glucose", options=["Normal", "Above Normal", "High"])
            smoke = col1.toggle("Do you smoke?")
            alco = col2.toggle("Do you consume alcohol?")
            active = col1.toggle("Are you physically active?", value=True)

    # Data Preprocessing
    gender_val = 1 if gender == "Female" else 2 
    chol_map = {"Normal": 1, "Above Normal": 2, "High": 3}
    bmi = weight / ((height / 100) ** 2)
    
    input_data = [[age*365, gender_val, height, weight, ap_hi, ap_lo, 
                   chol_map[chol], chol_map[gluc], int(smoke), int(alco), int(active), bmi]]

    st.markdown("---")
    if st.button("🔍 Analyze My Risk"):
        try:
            model = joblib.load("cardio_model.pkl")
            scaler = joblib.load("scaler.pkl")
            
            scaled_input = scaler.transform(input_data)
            prob = model.predict_proba(scaled_input)[0][1]
            
            # Prediction Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e293b"},
                    'steps': [
                        {'range': [0, 30], 'color': "#22c55e"},
                        {'range': [30, 70], 'color': "#eab308"},
                        {'range': [70, 100], 'color': "#ef4444"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge)

            if prob > 0.5:
                st.error("### ⚠️ High Risk Detected")
                st.write("Your metrics suggest a higher probability of cardiovascular issues. Please consult a doctor.")
            else:
                st.success("### ✅ Low Risk Detected")
                st.write("Your metrics look healthy! Keep maintaining a balanced lifestyle.")
                
        except:
            st.warning("Please train and save the model in the 'Training' tab first!")