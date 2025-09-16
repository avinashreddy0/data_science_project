# loading all modules accordind to you projects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
import joblib

#loading data set 

df = pd.read_csv(r"C:\Users\indur\Downloads\job_recommendation_dataset.csv")
print(df)

print('📂 successfully we loaded a data set AI-Powered Job Recommendation System ')

#Exploaratory Data Analysis

print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.describe())
print(df.head())

print('sucessfully show Exploratory Data Analysis')


#handling data set LIKe missing values and duplicates in AI-Powered Job Recommendation System.

missing_values = df.dropna(subset=['Job Title','Company','Skills Required','Description'])
duplicates_values = df.drop_duplicates()

#labelencoder# Combine text columns
df['combined_text'] = (
    df['Company'].astype(str) + " " +
    df['Skills Required'].astype(str) + " " +
    df['Description'].astype(str)
)

# Count Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_text'])

# Encode target (Job Title)
labels = LabelEncoder()
y = labels.fit_transform(df['Job Title'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)
y_pred_titles = labels.inverse_transform(y_pred)
print(y_pred_titles)

y_test_titles = labels.inverse_transform(y_test)

# Put into DataFrame
result = pd.DataFrame({
    'Actual': y_test_titles,
    'Predicted': y_pred_titles
})
print(result)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



joblib.dump(model, "job_recommendation_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(labels, "label_encoder.pkl")
