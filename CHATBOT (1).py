#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Model creation

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model & Scaler saved successfully!")


# In[2]:


# diabetes_chatbot.py

import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="HealthMate", layout="centered")

# Title
st.title(" Diabetes Prediction Chatbot")
st.write("Hello! I will ask you a few health questions and predict your **diabetes risk**.")

# Collect user input (chatbot style)
preg = st.number_input(" Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input(" Glucose Level (mg/dL)", min_value=0, step=1)
bp = st.number_input(" Blood Pressure (mm Hg)", min_value=0, step=1)
skin = st.number_input(" Skin Thickness (mm)", min_value=0, step=1)
insulin = st.number_input(" Insulin Level (IU/mL)", min_value=0, step=1)
bmi = st.number_input(" BMI", min_value=0.0, step=0.1)
dpf = st.number_input(" Diabetes Pedigree Function", min_value=0.0, step=0.01, format="%.2f")
age = st.number_input(" Age", min_value=1, step=1)

# Prediction button
if st.button("🔍 Predict My Risk"):
    # Prepare input
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1] * 100

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes!\n\nConfidence: {probability:.2f}%")
        st.info("👉 Please consult a doctor for further tests and advice.")
    else:
        st.success(f"✅ Low Risk of Diabetes.\n\nConfidence: {100-probability:.2f}%")
        st.balloons()

    # 🔹 Personalized Health Tips
    st.subheader(" Health Tips for You")
    if glucose > 140:
        st.write(" Your **glucose** level is high. Reduce sugar intake and get regular check-ups.")
    if bmi > 30:
        st.write(" Your **BMI** suggests obesity. Try a balanced diet and exercise.")
    if bp > 120:
        st.write(" Your **blood pressure** is above normal. Reduce salt and manage stress.")
    if insulin > 200:
        st.write(" Your **insulin** level is high. This may indicate insulin resistance.")
    if age > 45:
        st.write(" Higher age increases diabetes risk. Regular screenings are recommended.")
    if bmi < 18.5:
        st.write(" Your BMI is low. Consider a nutritious diet to reach a healthy weight.")
    
    st.write("✅ Maintain a healthy lifestyle: exercise, eat balanced meals, and get regular checkups.")

