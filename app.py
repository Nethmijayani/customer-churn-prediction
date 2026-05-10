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
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# dashboard header and KPI

st.title("Customer Churn Prediction Dashboard")
st.markdown("Predict which customers are likely to leave and understand why")
st.markdown("---")

#KPI cards
total = len(df)
high_risk  = len(df[df["Risk_Level"] =="High Risk"])
med_risk   = len(df[df["Risk_Level"] == "Medium Risk"])
low_risk   = len(df[df["Risk_Level"] == "Low Risk"])
churn_rate = df["Churn"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{total:,}")
col2.metric("High Risk",       f"{high_risk:,}")
col3.metric("Medium Risk",     f"{med_risk:,}")
col4.metric("Low Risk",        f"{low_risk:,}")
col5.metric("Churn Rate",      f"{churn_rate:.1f}%")

st.markdown("---")

#Chart section
st.subheader(" Overview Charts")

col1, col2 = st.columns(2)

with col1:
    risk_counts = df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level" , "Count"]
    fig1 = px.pie(
        risk_counts,
        names ="Risk Level",
        values ="Count",
        title ="Customer Risk Distribution",
        color_discrete_map ={
            "High Risk": "#FF4B4B",
            "Medium Risk": "#FFA500",
            "Low Risk": "#00CC96"
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        df,
        x="Churn_Probability",
        nbins = 30,
        title="Churn Probability Distribution",

        color_discrete_sequence=["#636EFA"]
    )

    fig2.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)

#featue mportance chart
st.subheader("Top Factors Driving Churn")
importance_df= pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

fig3 =px.bar(
    importance_df,
    x="Importance" , y= "Feature",
    orientation="h",
    title ="Top 10 Most Important Features",
    color ="Importance",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig3, use_container_width=True)

