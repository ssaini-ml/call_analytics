#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


import nest_asyncio
import pandas as pd

from call_analytics.pii_utility import authenticate_pii_client, redact_pii_with_batches


# # 1 - PII Check

# ## Load data

# In[ ]:


data = pd.read_csv("data/pii test/ITOP-11532.csv", encoding="utf-8")
data.replace({pd.NA: "EMPTY"}, inplace=True)
data.info()


# In[ ]:


LANGUAGE_MAP = {
    ("pb", "139"): "it",
    ("pb", "140"): "de",
    ("pb", "540"): "de",
    ("pb_kosmeti", "140"): "de",
    ("pb_kosmeti", "540"): "de",
    ("pb-nem", "140"): "de",
    ("pb-nem", "540"): "de",
    ("pb at", "140"): "de",
    ("pb at", "540"): "de",
    ("pb_ch_de", "140"): "de",
    ("pb_ch_de", "540"): "de",
    ("the bee", "140"): "de",
    ("the bee", "540"): "de",
    ("pb-nl", "141"): "nl",
    ("pb_ch_fr", "140"): "fr",
    ("fr_pb", "140"): "fr",
    ("pb-fr", "141"): "fr"
}


# In[ ]:


data["language"] = data.apply(lambda x: LANGUAGE_MAP.get((x["warteschlang"], str(x["mandant"])), "de"), axis=1)
data["language"].value_counts()


# In[ ]:


data_de = data[data["language"] == "de"]
data_fr = data[data["language"] == "fr"]
data_it = data[data["language"] == "it"]
data_nl = data[data["language"] == "nl"]


# ### Redact PII
# Now we redact summaries, topics, and transcriptions. Sentiments will never include PII.

# In[ ]:


nest_asyncio.apply() # Required for running async functions in Jupyter notebooks

# German
client = await authenticate_pii_client()
subject_de = await redact_pii_with_batches(client, data_de.mail_subject, "de")
data_de["mail_subject_pii"] = subject_de

client = await authenticate_pii_client()
content_de = await redact_pii_with_batches(client, data_de.mail_content, "de")
data_de["mail_content_pii"] = content_de

# French
client = await authenticate_pii_client()
subject_fr = await redact_pii_with_batches(client, data_fr.mail_subject, "fr")
data_fr["mail_subject_pii"] = subject_fr

client = await authenticate_pii_client()
content_fr = await redact_pii_with_batches(client, data_fr.mail_content, "fr")
data_fr["mail_content_pii"] = content_fr

# Italian
client = await authenticate_pii_client()
subject_it = await redact_pii_with_batches(client, data_it.mail_subject, "it")
data_it["mail_subject_pii"] = subject_it

client = await authenticate_pii_client()
content_it = await redact_pii_with_batches(client, data_it.mail_content, "it")
data_it["mail_content_pii"] = content_it

# Dutch
client = await authenticate_pii_client()
subject_nl = await redact_pii_with_batches(client, data_nl.mail_subject, "nl")
data_nl["mail_subject_pii"] = subject_nl

client = await authenticate_pii_client()
content_nl = await redact_pii_with_batches(client, data_nl.mail_content, "nl")
data_nl["mail_content_pii"] = content_nl


# In[ ]:


data_out = pd.concat([data_de, data_fr, data_it, data_nl])
data_out.info()


# Let's store the DataFrame in csv format.

# In[ ]:


data_out.to_csv(
    "data/pii test/ITOP-11532_out.csv",
    index=False,
    encoding="utf-8",
)

