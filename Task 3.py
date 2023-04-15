#!/usr/bin/env python
# coding: utf-8

# # Task 3-CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[26]:


#improting the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# In[27]:


#importing the dataset from google drive
df = pd.read_csv('C:/Users/preet/Downloads/CarPrice_Assignment.csv')
df.head()


# In[28]:


#Checking for the null values in the dataset
df.isnull().sum()


# In[29]:


df.info()


# In[30]:


print(df.describe())


# In[31]:


df.CarName.unique()


# In[32]:


#So to predict the car price we take price column from the data set to predict 
import seaborn as sns


# In[33]:


#distribution of the price column
sns.set_style("dark")
plt.figure(figsize=(15, 10))
sns.distplot(df.price)
plt.show()


# In[34]:


plt.figure(figsize=(12, 10))
plt.title("Price histogram")
sns.histplot(x="CarName", data=df)
plt.show()


# In[9]:


#checking for the corelation of all the features of the cars form the dataset
print(df.corr())


# In[35]:


plt.figure(figsize=(20, 15))
correlations = df.corr()
sns.heatmap(correlations, cmap="Blues", annot=True)
plt.show()


# In[36]:


#Training the Car Price Prediction Model
predict = "price"
data = df[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[37]:


#Training the Car Price Prediction Model
#Spliting the dataset into training and testing 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[38]:


#Training the model and making predictions using the DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
print(predictions)


# In[39]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:




