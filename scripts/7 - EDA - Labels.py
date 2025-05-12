#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from call_analytics.settings import PATH_LABELED


# # 6 - Exploratory Data Analysis - Labels
# 
# In this notebook, we perform exploratory data analysis (EDA) on the labels assigned to a set of text documents. Since this is a multilabel classification problem, each document can have multiple labels, making it important to understand the distribution and co-occurrence of labels in the dataset.
# 
# Analyzing label distributions helps us identify:
# 
# - The most and least common labels, providing insights into dataset balance.
# - Potential imbalances, which can impact model performance.
# - How often labels appear together, revealing relationships between topics.
# 
# By visualizing and summarizing these distributions, we gain a better understanding of the dataset before applying machine learning techniques.
# 
# ### Objectives
# - Examine label frequency: Identify common and rare labels.
# - Analyze multilabel distributions: Understand how many labels each document has.
# - Explore label co-occurrence: Detect patterns in how labels appear together.

# ## Load Data

# In[ ]:


DEPARTMENT = "CS" # "CS" or "PS"

df_test = pd.read_csv(f"{PATH_LABELED}/labeled_train_{DEPARTMENT}.csv")
df_train = pd.read_csv(f"{PATH_LABELED}/labeled_test_{DEPARTMENT}.csv")
df = pd.concat([df_train, df_test])
df.info()


# We'll create an exploded version of the dataset here as well. It will contain one row per label, instead of one row per call.

# In[ ]:


df_labels = df.copy()

df_labels["label"] = df_labels["label"].apply(
    lambda x: [x for x in x.split(",")]
)
df_labels = df_labels.explode("label")
df_labels.info()


# ## Label Frequency
# Let's first have a look at how often each label occurs. Since it's a multilabel problem, this will show more labels than calls.

# In[ ]:


df_temp = df_labels.copy()
df_temp = df_temp.groupby("label").size().reset_index(name="count")
df_temp = df_temp.sort_values("count", ascending=False) 

# Plot
plt.figure(figsize=(10, 12))
sns.barplot(x='count', y='label', data=df_temp, palette=["black"]*len(df_labels["label"].unique()))

# Formatting
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Distribution in the Dataset')
plt.show()


# Let's do top-10 only

# In[ ]:


df_temp = df_labels.copy()
df_temp = df_temp.groupby("label").size().reset_index(name="count")
# Calculate percentage
total = df_temp["count"].sum()
df_temp = df_temp.sort_values("count", ascending=False)[:10]


df_temp["percent"] = df_temp["count"] / total * 100

# Plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='count', y='label', data=df_temp, palette=["black"]*len(df_temp))

# Annotate bars
for i, (count, percent) in enumerate(zip(df_temp["count"], df_temp["percent"])):
    ax.text(count / 2, i, f"{percent:.1f}%", color="white", ha='center', va='center', fontsize=12)

# Formatting
plt.xlabel('Frequency', fontsize=10)
plt.ylabel('Labels', fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Label Distribution in the Dataset')
plt.tight_layout()
plt.show()


# In[ ]:


df_temp["percent"].sum()


# ## Multilabel Distribution
# 
# Here we'll examine how many labels calls have.

# In[ ]:


df_temp = df.copy()
df_temp["label_count"] = df_temp["label"].apply(
    lambda x: len(x.split(","))
) 

df_temp.label_count.value_counts(normalize=True) * 100


# ## Label Co-occurrence
# 
# Let's have a look here at the top-20 most occurring label combinations.

# In[ ]:


df_temp = df.copy()
df_temp = df_temp[df_temp["label"].str.contains(",")]
df_temp["label_combination"] = df_temp["label"].apply(
    lambda x: [x for x in x.split(",")]
)
df_temp.label_combination.value_counts()[:20]

