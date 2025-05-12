#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('cd', '..')


# In[2]:


from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import pytz
import seaborn as sns

from call_analytics.settings import PATH_AIRCALL_PROCESSED


# # 2 - Exploratory Data Analysis - Aircall
# 
# This notebook aims to explore and understand the structure, distribution, and key characteristics of the call data retrieved from Aircall.

# In[3]:


### Specify the following variables ###
INPUT_FILE = f"{PATH_AIRCALL_PROCESSED}/20250101_20250224_PS.csv" # Just edit the filename. Directory is handled by the environment variable
#######################################


# ## Load data

# In[4]:


calls_df = pd.read_csv(INPUT_FILE, encoding="latin-1")


# ## Descriptive Statistics
# 
# ### Overview & Summary

# In[5]:


# Not all calls have a transcription, summary, topics, or sentiment. Replacing the PII artifact of that here with NA
calls_df.replace("Error: Document text is empty.", pd.NA, inplace=True)

# Converting the timestamp columns to datetime objects
for col in ["started_at", "answered_at", "ended_at"]:
    calls_df[col] = calls_df[col].apply(
        lambda x: (
            datetime.fromtimestamp(int(x), tz=timezone.utc).astimezone(
                pytz.timezone("Europe/Berlin")
            )
            if pd.notna(x)
            else pd.NA
        )
    )

calls_df["day_of_week"] = calls_df["started_at"].dt.day_name()
calls_df.day_of_week = pd.Categorical(
    calls_df.day_of_week,
    categories=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    ordered=True,
)

for ci_feature in ["transcription", "summary", "topics"]:
    # Using OpenAI rule of thumb of 1 token per 4 characters: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    calls_df[f"token_estimate_{ci_feature}"] = calls_df[ci_feature].apply(lambda x: len(x)/4 if pd.notna(x) else 0)

calls_df.info()


# What does the data look like?

# In[6]:


calls_df.head()


# ## Call Volume

# In[7]:


calls_df.pivot_table(
    values=["id", "recording", "transcription", "summary", "topics", "sentiment"],
    index=["number_name"],
    aggfunc={
        "id": "count",
        "recording": "count",
        "transcription": "count",
        "summary": "count",
        "topics": "count",
        "sentiment": "count",
    },
    observed=False,
)


# Note here that even though all calls have a recording, not all of them have a transcription, summary, sentiment, and topics.

# In[8]:


calls_per_day = calls_df.groupby(calls_df["started_at"].dt.date).size()

plt.figure(figsize=(10, 5))
calls_per_day.plot(kind='bar', color='black', edgecolor='black')

plt.xlabel("Date")
plt.ylabel("Number of Calls")
plt.title("Sampled Number of Calls Per Day")
plt.xticks(rotation=75)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# ## Call Duration

# In[9]:


calls_df.duration.describe()


# In[10]:


# Plot histogram of call durations
plt.figure(figsize=(10, 5))
plt.hist(calls_df['duration'], bins=20, color='black', edgecolor='white')

plt.xlabel("Call Duration (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Call Durations")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
sns.boxplot(x=calls_df["day_of_week"], y=calls_df["duration"], palette=["black"]*7, fill=False)

plt.xlabel("Call Duration (seconds)")
plt.ylabel("Duration")
plt.title("Boxplot of Call Durations by Day of the Week")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


# ## Conversational Intelligence

# In[12]:


calls_df[calls_df["token_estimate_transcription"] != 0]["token_estimate_transcription"].describe()


# In[13]:


calls_df[calls_df["token_estimate_transcription"] != 0]["token_estimate_transcription"].describe()


# In[14]:


calls_df[calls_df["token_estimate_summary"] != 0]["token_estimate_summary"].describe()


# In[15]:


calls_df[calls_df["token_estimate_topics"] != 0]["token_estimate_topics"].describe()

