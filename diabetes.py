#!/usr/bin/env python
# coding: utf-8

# In[23]:

import streamlit

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df=pd.read_csv("diabetes.csv")
print(df.head())
print(df.describe())
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




