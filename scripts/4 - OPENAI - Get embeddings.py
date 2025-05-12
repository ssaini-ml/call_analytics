#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('cd', '..')


# In[ ]:


import os

import pandas as pd

from call_analytics.openai_utility import authenticate_openai_client, get_embeddings
from call_analytics.settings import PATH_AIRCALL_PROCESSED, PATH_OPENAI_EMBEDDINGS


# # 3 - OpenAI - Get embeddings
# 
# This notebook retrieves text embeddings for call-related text features using OpenAI's embedding API (e.g., text-embedding-3-large). These embeddings represent the semantic content of text in high-dimensional vector space and are a critical component for downstream machine learning tasks.
# 
# The input text can be full transcriptions, summaries, or other derived textual representations (e.g., customer-only summaries). This notebook handles batching, API interaction, and storage of the resulting embeddings for further use in the modeling pipeline.

# In[ ]:


### Specify the following variables ###
INPUT_FILE = "20250101_20250224_PS_addedfeatures.csv" # Just edit the filename. Directory is handled below data/aircall/processed

## Feature to embed ##
# Choose between "transcription", "summary", and "topics" for original Aircall features.
# Choose between "transcription_customer_only", "summary_customer_only", and "summary_guided" for the processed Aircall features.
FEATURE_TO_EMBED="summary_customer_only"
######################

EMBEDDING_MODEL="text-embedding-3-large"
#######################################

os.makedirs(PATH_OPENAI_EMBEDDINGS, exist_ok=True)

INPUT_FILE_PATH=f"{PATH_AIRCALL_PROCESSED}/{INPUT_FILE}"
OUTPUT_FILE_PATH=f"{PATH_OPENAI_EMBEDDINGS}/{INPUT_FILE.replace('.csv', '')}_{FEATURE_TO_EMBED}_embeddings_{EMBEDDING_MODEL}.csv"


# ## Load Data

# In[ ]:


calls_df = pd.read_csv(INPUT_FILE_PATH, encoding="latin-1")


# ## Remove Empty Rows

# In[ ]:


# Not all calls have a transcription, summary, topics, or sentiment. Replacing the PII artifact of that here with NA
calls_df.replace("Error: Document text is empty.", pd.NA, inplace=True)

calls_df = calls_df[["id", FEATURE_TO_EMBED]]
calls_df.dropna(subset=[FEATURE_TO_EMBED], inplace=True)


# In[ ]:


calls_df.info()


# ## Get Embeddings

# In[ ]:


client = authenticate_openai_client()

calls_df["embeddings"], calls_df["usage"] = zip(*calls_df[FEATURE_TO_EMBED].apply(lambda x: get_embeddings(client=client, text=x, model=EMBEDDING_MODEL)))

# Example: Assuming df["embeddings"] contains lists of 3072 floats
df_embeddings = pd.DataFrame(calls_df["embeddings"].apply(pd.Series))

# Rename columns to embedding_0, embedding_1, ..., embedding_3071
df_embeddings.columns = [f"embedding_{i}" for i in range(df_embeddings.shape[1])]

# Concatenate back with original DataFrame (optional)
calls_df = pd.concat([calls_df, df_embeddings], axis=1)

# Drop the original embeddings column (optional)
calls_df.drop(columns=[FEATURE_TO_EMBED, "embeddings"], inplace=True)

# embeddings, usage = get_embeddings(client=client, text="Hello world!", model=EMBEDDING_MODEL)
# print(len(embeddings), usage)


# In[ ]:


calls_df.info()


# In[ ]:


calls_df.head()


# In[ ]:


calls_df.usage.describe()


# ## Store Embeddings

# In[ ]:


calls_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8")

