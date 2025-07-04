#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import df


# In[15]:


#df_new=df[(df[1]!=0) & (df[2]!=0) & (df[3]!=0) & (df[5]!=0)]
df_new = df[(df[['Glucose','BloodPressure','SkinThickness','BMI']] != 0).all(axis=1)]


# In[16]:


X=df_new.drop(columns=['Outcome']).to_numpy()
Y=df_new['Outcome'].to_numpy()


# In[17]:


print("The shape of X: ",X.shape)
print("The shape of Y: ",Y.shape)


# In[18]:
X_poly = np.hstack([X, X**2, X**3, X**4, X**5])

#Feature Scaling
nu=np.mean(X,axis=0)
X_norm=(X_poly-(nu.T))/(np.max(X_poly,axis=0)-np.min(X_poly,axis=0))


# In[19]:


#Training set split
X_train=X_norm[:int(0.6*532),:]
Y_train=Y[:int(0.6*532)]
X_cv=X_norm[int(0.6*532):int(0.8*532),:]
Y_cv=Y[int(0.6*532):int(0.8*532)]
X_test=X_norm[int(0.8*532):,:]
Y_test=Y[int(0.8*532):]

