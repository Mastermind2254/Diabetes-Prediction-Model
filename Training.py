#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import df


# In[119]:


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
nu=np.mean(X_poly,axis=0)
X_norm=(X_poly-(nu.T))/(np.max(X_poly,axis=0)-np.min(X_poly,axis=0))
range_poly=np.max(X_poly,axis=0)-np.min(X_poly,axis=0)

# In[19]:


#Training set split
X_train=X_norm[:int(0.6*532),:]
Y_train=Y[:int(0.6*532)]
X_cv=X_norm[int(0.6*532):int(0.8*532),:]
Y_cv=Y[int(0.6*532):int(0.8*532)]
X_test=X_norm[int(0.8*532):,:]
Y_test=Y[int(0.8*532):]


# In[85]:


def sigmoid(z):
    f_wb=(1/(1+np.exp(-z)))
    return f_wb


# In[86]:


def cost_func(X_train,Y_train,W,b,lambda_):
    m,n=X_train.shape
    cost=0
    for i in range(m):
        z=np.dot(W,X_train[i,:])+b
        f_wb=sigmoid(z)
        cost+=Y_train[i]*np.log(f_wb)+(1-Y_train[i])*np.log(1-f_wb)
    reg=(lambda_/(2*m))*np.sum(W**2)
    cost=(-1/m)*cost
    cost=cost+reg
    return cost


# In[87]:


def gradient(X_train,Y_train,W,b,lambda_):
    m=X_train.shape[0]
    dw=0
    db=0
    for i in range(m):
        z=np.dot(W,X_train[i,:])+b
        f_wb=sigmoid(z)
        dw+=(f_wb-Y_train[i])*X_train[i,:]
        db+=(f_wb-Y_train[i])
    dw=((1/m)*dw)+(lambda_/m)*W
    db=(1/m)*db
    return dw,db


# In[88]:


def model(X_train,Y_train,W,b,alpha,lambda_,it):
    Wf=W
    bf=b
    for i in range(it):
        dw1,db1=gradient(X_train,Y_train,Wf,bf,lambda_)
        Wf=Wf-(alpha*dw1)
        bf=bf-(alpha*db1)
    return Wf,bf


# In[108]:


W=np.zeros(X_poly.shape[1])
b=0
alpha=0.01
iterations=10000
lambda_=0.01
predict=0
W_new,b_new=model(X_train,Y_train,W,b,alpha,lambda_,iterations)


# In[109]:


m_train=X_train.shape[0]
m_cv=X_cv.shape[0]
m_test=X_test.shape[0]
z_train=np.dot(W_new,X_train.T)+b_new
Y_hat_train=sigmoid(z_train)
z_cv=np.dot(W_new,X_cv.T)+b_new
Y_hat_cv=sigmoid(z_cv)
z_test=np.dot(W_new,X_test.T)+b_new
Y_hat_test=sigmoid(z_test)
J_train=(-1/m_train)*np.sum(Y_train*np.log(Y_hat_train)+(1-Y_train)*np.log(1-Y_hat_train))
J_cv=(-1/m_cv)*np.sum(Y_cv*np.log(Y_hat_cv)+(1-Y_cv)*np.log(1-Y_hat_cv))
J_test=(-1/m_test)*np.sum(Y_test*np.log(Y_hat_test)+(1-Y_test)*np.log(1-Y_hat_test))
print("J_train: ",J_train)
print("J_cv: ",J_cv)
print("J_test: ",J_test)
print("The cost: ",cost_func(X_train,Y_train,W_new,b_new,lambda_))


# In[129]:


'''user_input=input("Enter the feature list: ")
X_new=np.array([float(i) for i in user_input.split(',')])
X_poly_new = np.hstack([X_new, X_new**2, X_new**3, X_new**4, X_new**5])
X_norm=(X_poly_new-(nu.T))/(np.max(X_poly,axis=0)-np.min(X_poly,axis=0))
z=np.dot(W_new,X_poly_new.T)+b_new
Y_hat=sigmoid(z)
if Y_hat>=0.5:
    predict=1
elif Y_hat<0.5:
    predict=0
print(Y_hat)
print(f"Prediction: {predict}")'''

user_input = input("Enter the feature list (comma separated): ")
X_new = np.array([float(i) for i in user_input.split(',')]).reshape(1, -1)

# Polynomial expansion
X_poly_new = np.hstack([X_new, X_new**2, X_new**3, X_new**4, X_new**5])

# Normalize with training stats
X_norm = (X_poly_new - nu) / (range_poly + 1e-8)

# Prediction
z = np.dot(X_norm, W_new) + b_new
Y_hat = sigmoid(z)
predict = Y_hat >= 0.5

print(f"Probability: {Y_hat.item():.4f}")
print(f"Prediction: {'Diabetic' if predict == 1 else 'Non-Diabetic'}")


# In[ ]:





# In[ ]:




