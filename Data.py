#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('diabetes.csv')

if __name__=='__main__':
    df.head()

    print(df.dtypes)
    print("\nShape = ",df.shape)

    print(df.info())

    print(df.describe())

    df.isnull().sum()

    #df_new=df[(df[1]!=0) & (df[2]!=0) & (df[3]!=0) & (df[5]!=0)]
    df_new = df[(df[['Glucose','BloodPressure','SkinThickness','BMI']] != 0).all(axis=1)]
    X=df_new.drop(columns=['Outcome']).to_numpy()
    Y=df_new['Outcome'].to_numpy()

    #Histogram of all features vs number of examples
    try:
        for i in list(df.columns):
            plt.hist(X[:,df.columns.get_loc(i)],bins=27,edgecolor='black')
            plt.xlabel(f"Range of {i}")
            plt.ylabel("Number of people")
            plt.show()
    except IndexError:
        pass

    #Scatter Plot of two major features
    plt.scatter(X[Y==0][:,1],X[Y==0][:,5])
    plt.scatter(X[Y==1][:,1],X[Y==1][:,5])
    plt.xlabel('Glucose Level')
    plt.ylabel('BMI')
    plt.show()

