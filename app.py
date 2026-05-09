import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap 
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title= "Churn Prediction Dashboard",
    page_icon="🔮",
    layout= "wide"
)

#load saved model and feature columns
@st.cache_resource
def load_model():
    with open("model/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
        return model, features

model, feature_columns = load_model()        
