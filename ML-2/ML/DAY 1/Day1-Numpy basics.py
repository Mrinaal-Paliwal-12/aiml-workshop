#!/usr/bin/env python
# coding: utf-8

# # NUMPY BASICS
# 

# In[1]:


import numpy as np


# In[2]:


x=[1,2,3,4,5,6]


# In[3]:


x


# In[4]:


type(x)


# In[5]:


y=np.array([1,2,3,4,5,6])


# In[6]:


type(y)


# In[7]:


y=np.array((1,2,3,4,5,6))


# In[8]:


type(y)


# In[9]:


y


# In[10]:


x=[1,2,'vaibhav',69]
x


# In[11]:


y=np.array(x)
y


# In[12]:


y=np.linspace(start=0,stop=20,num=10) 


# In[13]:


y


# In[14]:


y=np.linspace(start=0,stop=20,num=10,endpoint=True)
y


# In[15]:


y=np.linspace(start=0,stop=20,num=10,endpoint=False)
y


# In[16]:


y=np.linspace(start=0,stop=20,num=10,endpoint=True,retstep=True) #retstrp gives the step if True
y


# In[17]:


y=np.linspace(start=0,stop=20,num=10,endpoint=True,retstep=False)
y


# In[18]:


d=np.arange(start=1,stop=10,step=2) #arange is to create array with conditions 
d


# In[19]:


# 3D Array 
#it is giving a normal nested list 
z=[[1,2,3],[4,5,6],[7,8,9]]
z


# In[20]:


z=np.array([[1,2,3],[4,5,6],[7,8,9]]) #if we use np.array it will give 3D array
z


# In[21]:


# array of 1's 
# give the parameter 
np.ones((3,3)) #means 3x3 matrix of one's


# In[22]:


# now with zero array 
np.zeros((4,5)) # means 4x5 matrix array of zeroes


# In[23]:


np.eye(4) # for identity matrix , means diagonal is 1 


# # RESHAPING

# In[24]:


x=np.array([0,1,2,3,4,5,6,7,8,9])
x


# In[25]:


x.reshape((5,2)) # change the shape of 1D matrix to given parameters


# # SLICING

# In[26]:


a=np.arange(10) # it will create array


# In[27]:


a


# In[28]:


s= slice(2,7,2)
# start , end , step


# In[29]:


a[s]


# In[30]:


s=slice(1,6,2)
a[s]


# In[31]:


x[2:7]


# In[32]:


x[-6:-2]


# # random number generation 

# In[33]:


np.random.rand(5) #gives random numbers 


# In[34]:


np.random.rand(5,4)  #gives random numbers in 2D format if parameters are given


# In[35]:


np.random.randn(5) #


# In[36]:


np.random.randn(5,5)


# # randint

# In[37]:


np.random.randint(low=1,high=100)


# In[38]:


np.random.randint(low=1,high=100,size=20)


# # seed 

# In[39]:


np.random.seed(69) # seed is for picking same value of random data 
#69 is the starting point 
# 4 is the random 4 numbers from starting point 
np.random.rand(4)


# In[40]:


np.random.seed(69)
np.random.rand(5)


# # Broadcasting
# 

# In[41]:


#broadcasting means the things we will do to the array it will be reflected to whole array 
arr=np.arange(0,10,1)
arr


# In[42]:


arr1=arr/10 #here we are div the whole array by 10 
arr1


# In[43]:


arr*5 # here we are multiplying it by 5


# In[44]:


slice_arr=arr[0:6]
slice_arr


# In[45]:


slice_arr[:]=555
slice_arr


# In[46]:


arr_copy = arr.copy()


# In[47]:


arr_copy[:]=1000
arr_copy


# In[48]:


arr


# # Conditional Selection
# 

# In[49]:


arr=np.arange(0,10,1)
arr


# In[50]:


boo_arr=arr>4 # it is checking the bool value with the condtion provided if the condition is True the 
#will return True else False
boo_arr


# In[51]:


arr[arr>4]


# In[52]:


arr


# In[53]:


arr+arr # array + array it will return the point wise array


# In[54]:


arr * arr


# In[55]:


arr/arr # warning becoz of the zero , that's why the first value is Nan (not a number)


# In[56]:


arr**9 # ** is raise  


# # Universal Array Functions 

# In[57]:


arr


# In[58]:


np.sqrt(arr) # sq root of each number


# In[59]:


np.exp(arr)


# In[60]:


np.sin(arr) 


# In[61]:


size = 100


# In[62]:


x1=range(size)
y1=range(size)


# In[63]:


x1


# In[64]:


y1


# In[65]:


get_ipython().run_cell_magic('timeit', '# it will give the time to execute this statement', '[x1[i]*y1[i] for i in range(size)]')


# In[66]:


x2=np.arange(size)
y2=np.arange(size)


# In[67]:


get_ipython().run_cell_magic('timeit', '', 'x2*y2')


# # Append

# In[68]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a


# In[69]:


a_row=np.append(a,[[21,22,69]],axis=0) #axis = 0 means adding a row 
a_row                                  # axis = 1 means adding a coloum


# In[70]:


a_col=np.array([34,23,55]).reshape(3,1)
a_col


# In[71]:


a_col=np.append(a,a_col,axis=1)
a_col


# In[ ]:




