import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 🔹 Create dataset inside code
data = {
    "fever": [1,1,0,1,1,0,1,0,1,0],
    "cough": [1,1,1,0,0,1,0,0,1,0],
    "headache": [0,1,1,1,0,0,1,0,1,0],
    "sore_throat": [1,0,0,0,0,1,1,0,1,0],
    "fatigue": [1,1,0,1,1,0,0,0,1,0],
    "disease": ["Flu","COVID-19","Common Cold","Migraine","Typhoid",
                "Common Cold","Migraine","Typhoid","Flu","Common Cold"]
}
df = pd.DataFrame(data)

# Training
X = df.drop("disease", axis=1)
y = df["disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# UI
st.title("🤖 Mini AI Doctor")
st.write("Enter your symptoms and let AI guess your disease!")
st.write(f"Model Accuracy: **{acc*100:.2f}%**")

# Sidebar Inputs
st.sidebar.header("Your Symptoms")
fever = st.sidebar.checkbox("Fever")
cough = st.sidebar.checkbox("Cough")
headache = st.sidebar.checkbox("Headache")
sore_throat = st.sidebar.checkbox("Sore throat")
fatigue = st.sidebar.checkbox("Fatigue")

# Input
input_data = pd.DataFrame([{
    "fever": int(fever),
    "cough": int(cough),
    "headache": int(headache),
    "sore_throat": int(sore_throat),
    "fatigue": int(fatigue)
}])

# Prediction
prediction = model.predict(input_data)[0]
st.subheader("Prediction Result")
st.write("🩺 The AI Doctor suggests you may have: **", prediction, "**")
