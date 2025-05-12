#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('cd', '..')


# In[2]:


import re

import pandas as pd
from tqdm import tqdm

from call_analytics.openai_utility import authenticate_openai_client, get_guided_summary
from call_analytics.settings import PATH_AIRCALL_PROCESSED


# # 3 - OpenAI - Get additional features
# 
# This notebook is used to generate new text-based features for call data, with a focus on extracting or transforming the raw transcription into more structured or targeted representations.
# 
# In particular, the feature generated here is a derived textual summary that captures only what the customer said during the call. This representation is useful for downstream machine learning tasks such as intent classification, topic tagging, or trend analysis, where isolating the customer's perspective can reduce noise and improve signal quality.
# 
# The goal of this process is to create concise, focused inputs that reflect the customer's expressed needs, concerns, or questions — enabling more robust modeling and interpretation in subsequent stages of the pipeline.

# In[3]:


### Specify the following variables ###
INPUT_FILE = "20250101_20250224_PS.csv" # Just edit the filename. Directory is handled below data/aircall/processed
#######################################

INPUT_FILE_PATH=f"{PATH_AIRCALL_PROCESSED}/{INPUT_FILE}"
OUTPUT_FILE_PATH=f"{PATH_AIRCALL_PROCESSED}/{INPUT_FILE.replace('.csv', '_addedfeatures.csv')}"


# ## Load Data

# In[4]:


calls_df = pd.read_csv(INPUT_FILE_PATH, encoding="latin-1")


# In[5]:


def remove_agent_transcription(transcription: str) -> str:
    """
    Removes all agent utterances from a call transcription, returning only the parts spoken by the customer.

    This function assumes that the transcription text includes speaker tags in the format "AGENT_<id>:" and "CUSTOMER:".
    It removes all lines and segments spoken by any agent, preserving only the customer's speech as plain text.

    Args:
        transcription (str): The full call transcription containing both agent and customer speech.

    Returns:
        str: A cleaned string containing only the customer's speech, with agent content removed.
    """
    text = re.sub(r"(AGENT_\d:.*?)(?:CUSTOMER:)", "CUSTOMER:\n", transcription, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"(AGENT_\d:.*)", "", text, flags=re.DOTALL | re.MULTILINE)
    text = text.replace("CUSTOMER:\n\n", "")
    text = text.replace("CUSTOMER:\n", "")
    text = text.replace("CUSTOMER:", "\n")
    return text

calls_df["transcription_customer_only"] = calls_df["transcription"].apply(remove_agent_transcription)


# ## Get Summary

# In[6]:


client = authenticate_openai_client()

file = open("data/prompts/customerOnlySummary.txt", "r")
instructions = file.read()
file.close()

customer_only_summaries = []

for text in tqdm(calls_df["transcription_customer_only"]):
    customer_only_summaries.append(get_guided_summary(client, instructions, text))

calls_df["summary_customer_only"] = customer_only_summaries


# ## Store Output

# In[7]:


calls_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8")

