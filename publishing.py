#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


# In[2]:


pd = pd.read_csv("extract.csv", header = None)
pd


# In[3]:


pd_task1 = pd[["payment", "year"]]
pd_task1 = pd_task1.dropna()
pd_task1[["payment"]] = pd_task1[["payment"]].replace("[\$,]", "", regex = True).astype(float)
pd_task1 = pd_task1[(pd_task1["payment"] > 0) & (pd_task1["payment"] <= 1000000)]
pd_task1[["year"]] = pd_task1[["year"]].replace("2017, 2019", "2017", regex = True)
pd_task1[["year"]] = pd_task1[["year"]].replace("2020 release", "2020", regex = True)
pd_task1[["year"]] = pd_task1[["year"]].astype(int)
pd_task1 = pd_task1.sort_values(by = ["year"], ascending = True)
pd_task1


# In[4]:


pd_task1_x = pd_task1["payment"]
pd_task1_y = pd_task1["year"]


# In[5]:


plt.scatter(pd_task1_x, pd_task1_y)
plt.xlabel("Year")
plt.ylabel("Payment (in millions)")
plt.show()


# In[6]:


pd_task1_model = st.linregress(pd_task1_x, pd_task1_y)
pd_task1_model

pd_task2a = pd[["payment", "race"]]
print(pd_task2a)