#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("creditcard.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# # FRAUD CASES AND GENUINE CASES

# In[6]:


fraud_cases=len(df[df['Class']==1])


# In[7]:


print(' Number of Fraud Cases:',fraud_cases)


# In[8]:


non_fraud_cases=len(df[df['Class']==0])


# In[9]:


print('Number of Non Fraud Cases:',non_fraud_cases)


# In[11]:


fraud=df[df["Class"]==1]


# In[13]:


no_fraud=df[df["Class"]==0]


# In[14]:


len(fraud)


# In[15]:


len(no_fraud)


# In[16]:


fraud.Amount.describe()


# In[17]:


no_fraud.Amount.describe()


# # EDA

# In[19]:


df.hist(figsize=(20,20),color='blue')
plt.show()


# In[22]:


from pylab import rcParams
rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(no_fraud.Time, no_fraud.Amount)
ax2.set_title('No Fraud')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# # CORRELATION

# In[23]:


plt.figure(figsize=(10,8))
corr=df.corr()
sns.heatmap(corr,cmap='BuPu')


# # Let us build our models:

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X=df.drop(['Class'],axis=1)


# In[36]:


y=df['Class']


# In[58]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=123)


# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


rfc=RandomForestClassifier()


# In[ ]:


model=rfc.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,prediction)


# # Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[48]:


model=lr.fit(X_train,y_train)


# In[49]:


prediction=model.predict(X_test)


# In[50]:


accuracy_score(y_test,prediction)


# # Decision Tree

# In[54]:


from sklearn.tree  import DecisionTreeRegressor
dt=DecisionTreeRegressor()


# In[55]:


model=dt.fit(X_train,y_train)


# In[56]:


prediction=model.predict(X_test)


# In[57]:


accuracy_score(y_test,prediction)


# # All of the models performed with a very high accuracy.

# In[ ]:




