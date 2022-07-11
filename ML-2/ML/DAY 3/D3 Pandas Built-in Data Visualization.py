#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df1=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df31.csv',index_col=0)
df2=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df32.csv')


# In[3]:


df1


# In[4]:



df2


# ### PLOT
# 
# 

# 
# ## Histograms

# In[6]:



df1['A'].plot.hist() #if we dont need 
#this <AxesSubplot:ylabel='Frequency'>
#then add semicolon at end


# In[7]:


df1['A'].plot.hist(); 


# In[8]:


df1['A'].plot.hist(edgecolor='k').autoscale(enable=True,axis='both') # k is for black


# In[9]:


df1['A'].plot.hist(edgecolor='k').autoscale(enable=True,axis='both',tight=True) 
# k is for black
#tight is for removing extra space


# In[10]:


df1['A'].plot.hist(bins=40,edgecolor='k').autoscale(enable=True,axis='both') 
#we can set bins also


# In[12]:


df1['A'].hist(); 


# In[13]:


df1['A'].hist(grid=False).set_ylabel("FREQUENCY"); #remove the grid and add y label


# # Bar Plots

# In[14]:


df2.plot.bar();


# In[15]:


df2.plot.bar(stacked=True);


# In[16]:


df2.plot.barh(); # barh for horizontal bar plot


# # Line Plot
# 

# In[18]:


df2.plot.line(y='a',figsize=(8,4),lw=2); #figsize is the size of figure
#lw line width 


# In[20]:


df2.plot.line(y=['a','b','c'],figsize=(12,4),lw=4); #figsize is the size of figure


# In[37]:


df2.plot.line(y='a',figsize=(8,4),lw=2,ls=':'); #ls can be change the dotted line etc


# In[38]:


ax=df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3,bbox_to_anchor=(1.0,0.1))


# # Area plot

# In[21]:


df2.plot.area();


# In[22]:


df2.plot.area(alpha=0.69); #alpha is for channging the transparency


# In[23]:


df2.plot.area(stacked=False,alpha=0.69); 


# # scatter plot

# In[24]:


df1.plot.scatter(x='A',y='B');


# In[27]:



df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm');


# 
# # matplotlib
#  

# In[31]:


import matplotlib.pyplot as plt
plt.scatter(df1['A'],df1['B'],c=df1['C'],cmap='coolwarm')
plt.colorbar().set_label('C')
plt.xlabel('A')
plt.ylabel('B')
plt.show()


# ### scatter plot with size marker
# 

# In[33]:


df1.plot.scatter(x='A',y='B',s=df1['C']*30);


# In[34]:


df1.plot.scatter(x='A',y='B',s=df1['C']*30,alpha=0.69);


# # Box Plot

# In[35]:


df2.describe


# In[36]:


df2.boxplot();

