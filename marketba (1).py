#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
data = pd.read_csv("trxns.csv")
head= data.head()
columns = data.columns
print(head)


# In[10]:


Ref = data.Ref.unique()
print("AVAILABLE TRANSACTIONS")
print(Ref)


# In[11]:


basket = (data.groupby(["Ref","Descr"])["Quantity"]).sum().unstack().reset_index().fillna(0).set_index('Ref')
basket


# In[12]:


def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1


# In[13]:


basket_encoded = basket.applymap(hot_encode)
basket = basket_encoded

print("ENCODED")
print(basket)


# In[14]:


frq_items = apriori(basket, min_support = 0.02, use_colnames = True)
print("Frequent Items")
print(frq_items)


# In[15]:


rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules.head()

