#!/usr/bin/env python
# coding: utf-8

# ### Vaibhav Kumar
# ### Roll No : 19

# ### Multivarient Regression - mpg dataset

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder


# In[46]:


data=pd.read_fwf("D:\\vk\\TRIM 3\\ML\\DATASET\\auto-mpg.data")
data


# In[47]:


col_names=['mpg','cylinder','displacement','horsepower','weight','acceleration','modelyear','origin','carname']


# In[48]:


data.columns=col_names


# In[49]:


data


# In[50]:


data.head()


# In[51]:



data.describe()


# In[52]:


data.info()


# In[53]:


data.shape


# In[54]:


data.isnull()


# In[55]:


data['carname'].value_counts()


# In[56]:


type(data)


# In[57]:


sns.heatmap(data.isnull())


# **no null value is there**

# In[58]:


sns.countplot(x='origin',data=data)


# In[59]:


data['carname'].unique()


# In[60]:


data['carname']=[i[0]for i in data['carname'].str.split(' ')]


# In[61]:


data['carname'].unique()


# In[62]:


data['carname']=data['carname'].replace(['"chevrolet','"chevy','"chevroelt'],'chevrolet')
data['carname']=data['carname'].replace(['"volkswagen','"vokswagen','"vw'],'volkswagen')
data['carname']=data['carname'].replace('"maxda','mazda')
data['carname']=data['carname'].replace('"toyouta','toyota')
data['carname']=data['carname'].replace('"mercedes','mercedes-benz')
data['carname']=data['carname'].replace('"nissan','datsun')
data['carname']=data['carname'].replace('"capri','ford')
                
                                                 


# In[64]:


data['carname'].unique()


# In[65]:


org=pd.get_dummies(data.origin,prefix='org')
org


# In[66]:


sns.countplot(x='cylinder',data=data)


# In[67]:


cyl=pd.get_dummies(data.cylinder,prefix='cyl')
cyl


# In[68]:


data.info()


# In[69]:


sns.countplot(x='modelyear',data=data)


# In[70]:


year=pd.get_dummies(data.modelyear,prefix='year')
year


# In[71]:


cn=pd.get_dummies(data['carname'],prefix='cn')
cn


# In[72]:


data


# In[77]:


data.drop(['origin','cylinder','modelyear','carname'],axis=1,inplace=True)


# In[78]:


data


# In[74]:


data=pd.concat([data,cn,year,cyl,org],axis=1)
data



# In[79]:


data.shape


# In[84]:


data[['displacement','horsepower','weight','acceleration']]=StandardScaler().fit_transform(data[['displacement','horsepower','weight','acceleration']])


# In[81]:


sum(data.horsepower=='?')


# In[82]:


data=data[data.horsepower!='?']


# In[83]:


data


# In[85]:


y=data.pop('mpg')


# In[86]:


y


# In[87]:


x=data


# In[88]:


x.head(269)


# **dividing** 
# **80:20**
# 
# 

# In[89]:


trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2,random_state=69)


# In[90]:


LR=LinearRegression()
LR.fit(trainx,trainy)


# In[91]:


pred=LR.predict(testx)
mean_squared_error(pred,testy)


# In[93]:


df=pd.DataFrame({'Actual':testy,'Predicted':pred})


# In[94]:


df


# In[95]:


df1=df.head(25)
df1.plot(y=['Actual','Predicted'],kind='bar',figsize=(20,19));
plt.grid(which='major',ls='-',linewidth=0.5,color='g')
plt.grid(which='minor',ls=':',linewidth=0.5,color='b')
plt.show()


# In[96]:


print('Mean Absolute Error',mean_absolute_error(testy,pred))
print('Mean Squared Error',mean_squared_error(testy,pred))
print('Root Mean Sqaured Error',np.sqrt(mean_squared_error(testy,pred)))


# **divding 70:30**

# In[98]:


trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=69)


# In[99]:


LR=LinearRegression()
LR.fit(trainx,trainy)


# In[100]:


pred=LR.predict(testx)
mean_squared_error(pred,testy)


# In[101]:


df=pd.DataFrame({'Actual':testy,'Predicted':pred})
df


# In[102]:


df1=df.head(25)
df1.plot(y=['Actual','Predicted'],kind='bar',figsize=(20,19));
plt.grid(which='major',ls='-',linewidth=0.5,color='g')
plt.grid(which='minor',ls=':',linewidth=0.5,color='b')
plt.show()


# In[103]:


print('Mean Absolute Error',mean_absolute_error(testy,pred))
print('Mean Squared Error',mean_squared_error(testy,pred))
print('Root Mean Sqaured Error',np.sqrt(mean_squared_error(testy,pred)))


# **dividing 90:10**

# In[104]:


trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.1,random_state=69)


# In[105]:


LR=LinearRegression()
LR.fit(trainx,trainy)


# In[106]:


pred=LR.predict(testx)
mean_squared_error(pred,testy)


# In[107]:


df=pd.DataFrame({'Actual':testy,'Predicted':pred})
df


# In[108]:


print('Mean Absolute Error',mean_absolute_error(testy,pred))
print('Mean Squared Error',mean_squared_error(testy,pred))
print('Root Mean Sqaured Error',np.sqrt(mean_squared_error(testy,pred)))


# **80:20 :::** 
# Mean Absolute Error 2.012801621835443
# Mean Squared Error 6.538767175795156
# Root Mean Sqaured Error 2.5571013229426707
# 
# 
# 
# **90:10 :: **
# Mean Absolute Error 1.8791406249999998
# Mean Squared Error 6.1197006835937495
# Root Mean Sqaured Error 2.473802878887837
# 
# 
# 
# **70:30**
# Mean Absolute Error 1054299011303.7745
# Mean Squared Error 3.0270984824050786e+25
# Root Mean Sqaured Error 5501907380540.933
# 
# 

# **most least error is in 90 :10 split**

# In[ ]:




