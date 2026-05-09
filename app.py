import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide"
)

# Load saved model and feature columns
@st.cache_resource
def load_model():
    with open("model/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_columns = load_model()

@st.cache_data
def load_data():
    df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")
    
    # Clean data same way as training
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

df = load_data()
print(df.columns)

# Get predictions for all customers
X = df[feature_columns]
df["Churn_Probability"] = model.predict_proba(X)[:, 1]
df["Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.4, 0.7, 1.0],
    labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
)