import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,roc_auc_score
from xgboost import XGBClassifier
import pickle
import os

#load data
df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print ("\nMissing values:")
print(df.isnull().sum())

#clean data

#fill missing numerical values with median
num_cols =df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#fill missing categorical values with mode
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

#encode categorical variables
le =LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#split features and target---> churn is the target. so 1 = churned, 2= stayed
x= df.drop(columns =["Churn", "CustomerID"],errors= "ignore")
y = df["Churn"]

print("\nFeatures used:",list(x.columns))
print("Target distribution:\n", y.value_counts())

#train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state = 42, stratify= y
)

#train XGboost model
model = XGBClassifier(
    n_estimators =200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(x_train,y_train)

#evaluate model
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test,y_pred))
print(f"ROC-AUC Score:{roc_auc_score(y_test,y_prob):.4f}")

#save model and feature names
os.makedirs("model", exist_ok=True)

with open("model/churn_model_pkl","wb") as f:
    pickle.dump(model, f)

#save feature column names for use in app.py
with open("model/feature_columns.pkl", "wb") as f:
    pickle.dump(list(x.columns), f)

print("\nModel saved to model/churn_model.pkl")
print("Features saved to model/feature_columns.pkl")