#!/usr/bin/env python
# coding: utf-8

# In[81]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st


# In[82]:


pd = pd.read_csv("extract.csv")
pd


# In[83]:


pd_task2a = pd[["payment", "race"]]
pd_task2a = pd_task2a.dropna()
pd_task2a


# In[84]:


pd_task2a = pd_task2a[(pd_task2a["race"] == "Black") | (pd_task2a["race"] == "White")]
pd_task2a[["payment"]] = pd_task2a[["payment"]].replace("[\$,]", "", regex = True).astype(float)
pd_task2a = pd_task2a[(pd_task2a["payment"] > 10) & (pd_task2a["payment"] <= 1000000)]
pd_task2a


# In[85]:


pd_task2a_white = pd_task2a["race"] == "White"
pd_task2a_black = pd_task2a["race"] == "Black"


# In[86]:


pd_task2a_whitemean = pd_task2a.loc[pd_task2a_white, "payment"].mean()
pd_task2a_blackmean = pd_task2a.loc[pd_task2a_black, "payment"].mean()
print("The mean payment for white authors was: " + str(pd_task2a_whitemean))
print("The mean payment for black authors was: " + str(pd_task2a_blackmean))


# In[87]:


pd_task2a_2 = pd_task2a
pd_task2a_2["race"] = pd_task2a[["race"]].replace("Black", 1, regex = True)
pd_task2a_2["race"] = pd_task2a_2[["race"]].replace("White", 0, regex = True)
pd_task2a_2


# In[88]:


pd_task2a_model = st.linregress(pd_task2a_2["payment"], pd_task2a_2["race"])
pd_task2a_model


# In[89]:


chart_race = ["White", "Black"]
chart_mean = [pd_task2a_whitemean, pd_task2a_blackmean]

plt.bar(chart_race, chart_mean)
plt.xlabel("What race was the author?")
plt.ylabel("Payment")
plt.show()


# In[96]:


pd_task2b = pd[["payment", "race", "agent"]]
pd_task2b = pd_task2b[pd_task2b["race"] == "Black"]
pd_task2b = pd_task2b.dropna()
pd_task2b[["payment"]] = pd_task2b[["payment"]].replace("[\$,]", "", regex = True).astype(float)
pd_task2b = pd_task2b[(pd_task2b["payment"] > 10) & (pd_task2b["payment"] <= 1000000)]
pd_task2b


# In[97]:


task2b_agent = pd_task2b["agent"] == "Yes"
task2b_noagent = pd_task2b["agent"] == "No"

agent_mean = pd_task2b.loc[task2b_agent, "payment"].mean()
noagent_mean = pd_task2b.loc[task2b_noagent, "payment"].mean()
print("The mean payment for black authors with agents was: " + str(agent_mean))
print("The mean payment for black authors without agents was: " + str(noagent_mean))


# In[98]:


chart_race = ["Agent", "No Agent"]
chart_mean = [agent_mean, noagent_mean]

plt.bar(chart_race, chart_mean)
plt.xlabel("Did the black author have an agent?")
plt.ylabel("Payment")
plt.show()


# In[101]:


pd_task2b_final = pd_task2b[["payment", "agent"]]
pd_task2b_final


# In[103]:


pd_task2b_final["agent"] = pd_task2b_final[["agent"]].replace("Yes", 1, regex = True)
pd_task2b_final["agent"] = pd_task2b_final[["agent"]].replace("No", 0, regex = True)
pd_task2b_final


# In[105]:


pd_task2b_model = st.linregress(pd_task2b_final["payment"], pd_task2b_final["agent"])
pd_task2b_model


# In[ ]:




