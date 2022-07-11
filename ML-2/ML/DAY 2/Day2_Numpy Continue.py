#!/usr/bin/env python
# coding: utf-8

# # Day 2   Vaibhav Kumar , RollNo 19

# # Insert 

# In[1]:


import numpy as np


# In[2]:


b=np.array([[1,2,3],[4,5,6],[7,8,9]])
b


# In[3]:


b.shape
b.size


# In[4]:


b.shape


# In[5]:


b_ins=np.insert(b,1,[15,13,12],axis=0)#axis = 0 means adding a row    
#(array,row number,value,axis- row or coloum)                       
# axis = 1 means adding a coloum 
b_ins


# In[6]:


b_ins2=np.insert(b,2,[19,29,69],axis=1) # here pos = 2 means it will enter in 3rd coloumn
b_ins2


# # Matrix

# In[7]:


x=np.matrix('1,2,3;4,5,6;8,9,0') # seperated by semi colon
x


# In[8]:


x.shape


# In[9]:


x.size


# In[10]:


x[2,1]=-69 # if we want to change the matrix then we can directly call the element by position


# In[11]:


x


# In[12]:


x[1,:]


# # Matrix Multiplication 

# In[13]:


x*x # it will be proper matrix multiplication 


# In[14]:


# it will be point to point multiplication
b


# In[15]:


b*b


# In[ ]:




