#!/usr/bin/env python
# coding: utf-8

# ## Vaibhav Kumar
# ## Roll no : 19
# 

# #  Data INPUT AND OUTPUT
# ## CSV 
# 
# 

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df1.csv')
df


# In[5]:


df=pd.read_csv('df1.csv')
df #file is uploaded in jupiter home page 


# ### CSV OUTPUT

# In[6]:


df.to_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\example.csv',index=False)


# # excel
# 
# 

# In[7]:


df=pd.read_excel('D:\\vk\\TRIM 3\\ML\\DATASET\\Excel_Sample.xlsx',sheet_name='Sheet1')


# In[8]:


df


# In[9]:


df.drop('Unnamed: 0',axis=1,inplace=True)
df # dropping the coloumn


# ### excel output

# In[10]:


df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')


# ## Joining and Concatenating 

# In[11]:


df1=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df1.csv')
df1


# In[12]:


df2=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df2.csv')
df2


# In[13]:


df3=pd.read_csv('D:\\vk\\TRIM 3\\ML\\DATASET\\df3.csv')
df3


# ### Concatenating

# In[14]:


df4=pd.concat([df1,df2,df3])
df4 #it simply joining the given files one after another 


# In[17]:


df4.reset_index(inplace=True)
df4            


# In[18]:


df4.drop(['level_0'],axis=1,inplace=True) #removing a coloumn
df4


# In[19]:


pd.concat([df1,df2,df3],axis=1)#with column concat


# # joining

# In[22]:


left=pd.DataFrame({'A':['A0','A1','A2'],
                  'B':['B0','B1','B2']},
                 index=['K0','K1','K2'])

right =pd.DataFrame({'C':['C0','C2','C3'],
                     'D':['D0','D2','D3']},
                    index=['K0','K2','K3'])


# In[23]:


left.join(right)


# In[24]:


right.join(left)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




