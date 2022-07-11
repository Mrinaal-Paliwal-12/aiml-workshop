#!/usr/bin/env python
# coding: utf-8

# # Pandas  - Vaibhav Kumar , RollNo 19

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# create vector as a row
vector_row=np.array([1,2,3])


# In[3]:


# create vector as a column
vector_column=np.array([[1],[2],[3]])


# In[4]:


vector_row


# In[5]:


vector_column


# **Creating a matrix** 

# In[6]:


from scipy import sparse


# **only non zero elements will be stored in sparse matrix**

# In[7]:


matrix=np.array([[0,0], 
               [0,1],
               [3,0]])

#create compressed sparse row (CSR) Matrix
matrix_sparse = sparse.csr_matrix(matrix)


# In[8]:


matrix_sparse


# In[9]:


matrix


# In[10]:


print(matrix_sparse) # it is giving the position , means 1 is it location 1st row 1st col , 
# and 3 is in locaiton 2nd row 1st col 


# In[11]:


#create a large matrix
martix_large=np.array([[0,0,0,0,0,0,0,0],
                      [5,3,0,0,0,0,0,0],
                      [0,0,0,8,0,0,0,0],
                      [0,4,0,0,0,1,0,69]])


# In[12]:


#create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(martix_large)


# In[13]:


#view larger sparse matrix
print(matrix_large_sparse)


# In[14]:


#create 3x3 matrix
matrix=np.array([[1,2,3],
               [4,5,6],
                [7,6,9]])


# In[15]:


#finding maximum element in each column
np.max(matrix,axis=0)


# In[16]:


#finding maximum element in each row
np.max(matrix,axis=1)


# 

# In[17]:


#return mean
np.mean(matrix)


# In[18]:


#finding variance
np.var(matrix)


# In[19]:


#finding standard deviation
np.std(matrix)


# # Reshaping of Array

# In[20]:


#create 4x3 matrix
matrix=np.array([[1,2,3],
                [4,5,6],
                [6,7,8,],
                [6,54,7]])


# In[21]:


#reshaping matrix in 2x6 matrix
matrix.reshape(2,6)


# 

# In[22]:


matrix.reshape(1,-1)


# In[23]:


matrix.size


# In[24]:


matrix.reshape(matrix.size)


# In[25]:


#TRANSPOSE OF MATRIX
matrix.T


# In[26]:


#TRANSPOSE A VECTOR
np.array([1,2,3,4,5,6,7]).T


# In[27]:


#TRANSPOSE ROW VECTOR
np.array([[1,2,3,4,5,6,7]]).T


# In[28]:


#FLATTEN MATRIX
matrix.flatten()


# In[29]:


#Return Diagonal Elements
# need a NxN matrix to show result properly
#here we have 4x3 matrix that's why we are getting 3 elements only
matrix.diagonal()


# In[30]:


c= np.array([1,2,np.nan,4,5])
c # if we take nan value it will automaticly take floating point number


# In[31]:


np.isnan(c) # it will give nan value if there as TRUE other FALSE


# In[32]:


np.mean(c) 


# In[33]:


# if any data contains nan value and we need MEAN VALUE we need to use this 
a=np.mean(c[~np.isnan(c)])


# In[34]:


a


# In[35]:


matrix


# In[36]:


# astype() is to convert any datatype to another
a=matrix.astype('float')


# In[37]:


a


# # Important **********

# In[38]:


a[0][0]=np.nan


# In[39]:


a[1][1]=np.inf # infinity value


# In[40]:


a


# In[41]:


np.isinf(a) # checking for a infinity value


# In[42]:


np.isnan(a)


# In[44]:


flag=np.isinf(a)|np.isnan(a) # pipe fn is to check multiple conditions at a time 


# In[45]:


flag


# In[57]:


flag


# # DATA CLEANING

# In[46]:


a[flag]


# In[47]:


a[flag]=0 #replacing flags value to zero 
# not a right way to do


# In[48]:


a


# In[50]:


dict={'A':100,'B':200,'C':300,'D':400}
z1=pd.Series(dict)
z1


# In[54]:


z2=pd.Series([10,20,30,40],'A B D E'.split())


# In[55]:


'A B C D'.split()


# In[56]:


z2


# In[57]:


z1+z2


# # Slicing in Series
# 

# In[58]:


z1


# In[59]:


z1['A':'C']


# In[60]:


type(z1['A':'C'])


# In[61]:


z1[['A','C']]


# In[62]:


#using default index values
z1[[0,1]]


# ## Indexing

# In[63]:


# .iloc does integer based indexing 
z1.iloc[2] # need to specify the integer


# In[64]:


#loc does label based indexing
z1.loc['A'] # here we need to specify the label instead of integer


# 
# ## OPERATIONS
# 

# In[65]:


z1>100


# In[66]:


z1[z1>100] #condition


# In[67]:


b=[True,False,True,False]
z1[b]
# here where ever there is true it will return


# ### BROADCASTING

# In[68]:


z1+200


# ### Ordering on Series

# In[69]:


#sort_values()  func to perform sorting 
z1.sort_values(ascending=False)


# In[70]:


z1.sort_values(ascending=False).index[1]


# In[71]:


z1 # no change in data


# In[72]:


# if we want to change the data permanently use 
# inplace=True
z1.sort_values(ascending=False,inplace=True)


# In[73]:


z1


# ## aggregation on series
# 

# In[74]:


z1.mean()


# In[75]:


z1.sum()


# In[76]:


z1.max()


# In[77]:


z1.min() 


# In[79]:


z1.idxmax()


# ### Series with two Column

# In[80]:


data_1={'Column1':pd.Series([2,3,4,5],['a b c d'.split()]),
       'Column2':pd.Series([20,30,40,50],['A B C D'.split()])}


# In[81]:


data_1


# # DATA FRAME

# In[82]:


rand_mat=np.random.randn(5,4)
rand_mat


# In[83]:


df = pd.DataFrame(rand_mat,index='A B C D E'.split(),
columns = 'W X Y Z'.split())


# In[84]:


df


# ## Selection and Indexing

# In[85]:


df['W']


# In[86]:


df[['W','Z']]


# In[88]:


df[['W','X']]


# In[89]:


df.W #NOT RECOMENDDED 


# In[91]:


df.iloc[2]


# ### ADDING A ROW

# In[92]:


df.loc['F']=df.loc['A']+df.loc['B']


# In[93]:


df


# In[94]:


# if we want to reset the indexes as Default
df.reset_index()


# In[95]:


#for changing the index names
newindex = 'DEL UP UK TN AP KL'.split()


# In[96]:


df['States']=newindex # added a column


# In[97]:


df


# In[98]:


# now we have to replace abcdef to states values
df.set_index('States',inplace=True)


# In[99]:


df


# ### Multi -Index and Index Hierarchy

# In[103]:


# Index levels
outside=['North','North','North','South','South','South']
inside=newindex


# In[104]:


hier_index=list(zip(outside,inside))


# In[105]:


hier_index


# In[106]:


hier_index=pd.MultiIndex.from_tuples(hier_index)


# In[107]:


hier_index


# In[108]:


df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df


# In[109]:


# now we can fetch the data according to hierarcy 
df.loc['North']


# In[110]:


df.loc['North'].loc['DEL']


# In[111]:


#index not been assigned to any names
df.index.names


# In[112]:


df.index.names=['Region','States']


# In[113]:


#Return Cross-section from the series/dataframe
df.xs('North')


# In[115]:


outside=['North','North','North','South','South','South']
inside=[1,2,3,1,2,3]
hier_index=list(zip(outside,inside))
hier_index=pd.MultiIndex.from_tuples(hier_index)


# In[116]:


df=pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df.index.names=['Region','Num']
df


# In[118]:


df.xs(['North',1])


# In[119]:


df.xs(1,level='Num')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




