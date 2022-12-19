#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


wine_dataset = pd.read_csv('C:\\Users\\Pallavi\\Desktop\\MAJOR_PROJECT\\wine data.csv')


# In[3]:


#number of rows and columns in dataset
wine_dataset.shape


# In[4]:


wine_dataset.head()


# In[5]:


wine_dataset.isnull().any()


# In[6]:


wine_dataset.isnull().sum()


# In[7]:


wine_dataset[wine_dataset['total sulfur dioxide'].isnull()]


# In[8]:


wine_dataset[wine_dataset['pH'].isnull()]


# In[9]:


wine_dataset[wine_dataset['quality'].isnull()]


# In[10]:


wine_dataset.drop([9,123,184],axis=0,inplace=True)


# In[11]:


wine_dataset.isnull().sum()


# In[12]:


#statistical measures
wine_dataset.describe()


# In[13]:


sns.catplot(data = wine_dataset, x = 'quality', kind = 'count')


# In[14]:


#volatile acidity vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'volatile acidity')
plt.show()


# In[15]:


#citric acid vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'citric acid')
plt.show()


# In[16]:


#chlorides vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'chlorides')
plt.show()


# In[17]:


#free sulfur dioxide vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'free sulfur dioxide')
plt.show()


# In[18]:


#sulphates vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'sulphates')
plt.show()


# In[19]:


#alcohol vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(data=wine_dataset,x = 'quality',y = 'alcohol')
plt.show()


# # Correlation

# In[24]:


correlation = wine_dataset.corr()
plt.figure(figsize=(5,5))
plt.show()


# In[23]:


#construction of heatmap to understand correlation
sns.heatmap(correlation,cbar = True,square = True,fmt = '.1f',annot = True, annot_kws={'size':8},cmap = 'Blues')
plt.show()


# # Data Pre- Processing

# In[25]:


#seperating data and labels
X = wine_dataset.drop('quality',axis=1)


# In[26]:


Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0) #label binarization


# In[27]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# # Model Training Random Forest Classifier

# In[28]:


model = RandomForestClassifier()


# In[29]:


model.fit(X_train,Y_train)


# # Model Evaluation Accuracy Score

# In[30]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[31]:


print("Accuracy score:", test_data_accuracy)


# # Buiding A Predictive System

# In[32]:


input_data = (7,0.56,0.17,1.7,0.065,15,24,0.99514,3.52,0.68,10.55)
#changing to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshaping the data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
if prediction[0]==1:
    print('Good quality wine')
else:
    print('Bad quality wine')


# In[ ]:




