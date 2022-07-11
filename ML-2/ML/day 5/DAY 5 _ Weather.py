#!/usr/bin/env python
# coding: utf-8

# ### Vaibhav Kumar
# ### Roll No : 19

# ### Linear Regression , Weather Dataset , Weather Prediction Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[2]:


dfw=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\Weather.csv')


# In[3]:


dfw


# In[4]:


dfw.head()


# In[5]:



dfw.describe()


# In[6]:


dfw.shape


# In[7]:


x=dfw['MaxTemp']
y=dfw['MinTemp']


# In[8]:


plt.scatter(x,y)
plt.xlabel('MaxTemp',fontsize='12')
plt.ylabel('MinTemp',fontsize='12')
plt.show()


# In[9]:


sns.regplot(x,y,color='red')


# In[10]:


x.head()


# In[11]:


y.head()


# In[12]:


x.shape


# In[13]:


X_=x.values.reshape(-1,1)


# In[14]:


X_.shape


# In[15]:


x


# In[16]:


X_


# ### Model
# 

# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X_,y,test_size=0.2,random_state=30)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


LR=LinearRegression()
LR.fit(X_train,y_train)


# In[21]:


y_pred=LR.predict(X_test)


# In[22]:


y_test


# In[23]:


y_pred


# In[24]:


weights = LR.coef_
intercept = LR.intercept_
print(weights,intercept)


# In[25]:


plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred, color='green')
plt.show()


# In[26]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[27]:


df1=df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='red')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='green')
plt.show()


# In[28]:


print('Mean Absolute Error',mean_absolute_error(y_test,y_pred))
print('Mean Squared Error',mean_squared_error(y_test,y_pred))
print('Root Mean Sqaured Error',np.sqrt(mean_squared_error(y_test,y_pred)))


# **80:20** ::: 
# Mean Absolute Error 3.086071147707306
# Mean Squared Error 15.642509497194942
# Root Mean Sqaured Error 3.955061250751364
# 
# 
# **90:10** :: 
# Mean Absolute Error 3.1047256786542192
# Mean Squared Error 15.83161054325677
# Root Mean Sqaured Error 3.97889564367511
# 
# 
# **70:30**
# Mean Absolute Error 3.1013916713826886
# Mean Squared Error 15.768060731561533
# Root Mean Sqaured Error 3.970901752947501
# 

# **80:20 split has most least error compared to others**

# In[ ]:




