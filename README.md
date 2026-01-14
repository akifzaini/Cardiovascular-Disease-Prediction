## CARDIOVASCULAR DISEASE APP PREDICTION ##

A simple streamlit application that predict the risk of cardiovascular disease based on personal health information using the existing dataset "cleaned_cardio.csv".
This project also includes data exploration, visualization, model training, and real-time prediction in one interactive interface.

Dataset Source
Cardiovascular Disease Dataset (Kaggle): https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset.

This application allows users to:
- Explore a cleaned cardiovascular dataset
- Perform exploratory data analysis (EDA)
- Train a machine learning model (Logistic Regression)
- Test cardiovascular disease risk using personal health inputs

Features:

Home Page – Overview of the application

Dataset Exploration – Preview dataset and statistical summary

Data Analysis – Visual insights (distribution, boxplots, scatter plots)

Model Training – Train and evaluate Logistic Regression model

Health Risk Test – Predict CVD risk based on user input


Machine Learning Model
- Algorithm: Logistic Regression
- Preprocessing: StandardScaler
- Train/Test Split: 80% / 20%
- Evaluation Metrics:

The trained model and scaler are saved using joblib for reuse in prediction.

Health Risk Prediction
- Users can input their:
- Age, gender, height, weight
- Blood pressure values
- Cholesterol & glucose levels
- Lifestyle habits (smoking, alcohol, activity)

The app predicts:

✅ Low Risk of Cardiovascular Disease

⚠️ High Risk of Cardiovascular Disease

⚠️ This application is for educational purposes only and should not replace professional medical advice.


Type this command in your Visual Studio Code terminal to run the app

*streamlit run CardiovascularPrediction.py*
