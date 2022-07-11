#!/usr/bin/env python
# coding: utf-8

# ### Vaibhav Kumar 
# ### Roll no 19

# # DATA CLEANING

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # the data

# In[2]:


titanic=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\titanic.csv')


# In[3]:


titanic.head(10)


# In[4]:


titanic.info()


# ### Missing data

# In[5]:


titanic.isnull()


# In[9]:


sns.heatmap(titanic.isnull())


# **now we have to replace the age null value to some form of imputation** 
# 
# **and drop the cabin coloumn**

# In[10]:


sns.countplot(x='Embarked',data=titanic)


# 
# **From passenger list most of the passengers are in Embarked "S" class**

# In[11]:


sns.countplot(x='SibSp',hue='Pclass',data=titanic);


# *Color diff is the passengers class and x axis is siblings spouse*
# 
# **maximum passenegers are not dependent or zero dependance**

# In[12]:


titanic['Fare']


# In[13]:


titanic['Fare'].hist()


# **Max count comes in range 0-50**

# In[14]:


titanic['Gender']


# In[16]:


sns.countplot(x='Gender',data=titanic)


# **Males are more than Females**

# In[19]:


titanic['Age'].hist();


# **most of the Passengers are of age between 20-30**

# # Data Cleaning 

# In[20]:


titanic.describe()


# In[23]:


plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass',y='Age',data=titanic)


# In[24]:


titanic.groupby('Pclass').mean()['Age']


# In[28]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 41
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[29]:


titanic['Age']=titanic[['Age','Pclass']].apply(impute_age,axis=1)


# In[30]:


sns.heatmap(titanic.isnull())


# **Age has no missing value **

# In[31]:


titanic['Age']


# In[32]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[33]:


titanic.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




