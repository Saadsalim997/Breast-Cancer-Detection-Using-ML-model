#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
data = pd.read_csv('C:\\Users\\saads\\Downloads\\AI in Enterprise Lab 2\\wdbc.data', header=None)


# In[3]:


# Adding column names based on the dataset description
column_names = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
data.columns = column_names

# Mapping diagnosis to binary values
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Dropping the ID column
data = data.drop("ID", axis=1)


# In[4]:


# Splitting the dataset into features and target variable
X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[5]:


# Training a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




