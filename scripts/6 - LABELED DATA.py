#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

from call_analytics.settings import (
    PATH_MLSTUDIO_LABELED_DATA,
    PATH_AIRCALL_DATA,
    RANDOM_STATE,
    PATH_LABELED,
)


# # 5 - LABELED DATA
# 
# The purpose of this notebook is to combine labeled data from different sources. Part of the data was labeled by me (Luc Bams) in an Excel sheet for flexibility during label definition. The other part was labeled by SMEs in MLStudio with a fixed set of labels.
# 
# ## User Input

# In[ ]:


DEPARTMENT = "CS" # "CS" or "PS"

if DEPARTMENT == "CS":
    INPUT_EXCEL = f"{PATH_AIRCALL_DATA}/labeled/3225863507_2000_10_rs23_complete.xlsx"
    INPUT_MLSTUDIO = f"{PATH_MLSTUDIO_LABELED_DATA}/labeled_CS_0_1500.csv"
elif DEPARTMENT == "PS":
    INPUT_EXCEL = f"{PATH_AIRCALL_DATA}/labeled/PB_2000_0-500_rs23_complete_v2.xlsx"
    INPUT_MLSTUDIO = f"{PATH_MLSTUDIO_LABELED_DATA}/labeled_PS_0_750.csv"


# ## Load Data
# 
# ### MLStudio --> Labeled by Redcare
# 
# First we load the MLStudio data. This data is straightforward. We just select the columns we need and rename them. Instead of the url to the file used for labeling, we want the id of the call. The files were named using the call id, so that information is easy to extract. We will also replace the non-leaf node labels that were assigned by a leaf-node one (i.e. if an SME assigned, for example, Support/Order & Delivery we turn that into Support/Order & Delivery/Other, since they mean the same thing). This can happen since MLStudio allows non-leaf node labels. Semantically, Support/Order & Deliver means the same as Support/Order & Delivery/Other. It means it wasn't possible to be more specific than the first two levels.

# In[ ]:


df_mlstudio = pd.read_csv(INPUT_MLSTUDIO, usecols=["Url", "Label"])
df_mlstudio.rename(columns={"Url": "id", "Label": "label"}, inplace=True)
df_mlstudio["id"] = df_mlstudio["id"].apply(lambda x: x.split("/")[-1].replace(".txt", ""))
df_mlstudio.sort_values("id", inplace=True)
df_mlstudio.head()


# In[ ]:


print(f"MLStudio data shape: {df_mlstudio.shape}")


# #### Label check
# 
# Let's check whether we only have labels that we expect.
# 
# For CS, there are some non-leaf node labels. We change these into their corresponding leaf-node labels.

# In[ ]:


df_mlstudio["label"] = df_mlstudio["label"].apply(
    lambda x: [x for x in x.split(",")]
)
df_mlstudio = df_mlstudio.explode("label")

if DEPARTMENT == "CS":
    df_mlstudio["label"] = df_mlstudio["label"].apply(
        lambda x: (
            f"{x}/Other"
            if x
            in [
                "Informational/Order & Delivery",
                "Transactional/Order & Delivery",
                "Support/Payment & Discount",
                "Support/Order & Delivery",
            ]
            else x
        )
    )
elif DEPARTMENT == "PS":
    # No need to change the labels for PS. Only delete the label "CS" --> Not relevant for intent recognition
    df_mlstudio = df_mlstudio[df_mlstudio["label"] != "CS"]

print(len(df_mlstudio["label"].unique()))

# Uncomment below if you want to see the labels and how often they occur.
# Note that these numbers differ from the distributions later, since this takes all labels per call into account.
# Later on, we only take the first label into account.
# df_mlstudio["label"].value_counts(normalize=True)


# In[ ]:


# 'Implode' the data again, so that we have one row per call.
df_mlstudio = df_mlstudio.groupby("id")["label"].apply(lambda x: ",".join(x)).reset_index()
df_mlstudio.sort_values("id", inplace=True)
df_mlstudio.head()


# In[ ]:


print(f"MLStudio data shape: {df_mlstudio.shape}")


# ### Excel --> Labeled by Luc (Conclusion Intelligence)
# 
# This data needs some more processing, since this file was used during the early stages. There's a lot of information in there that's of no use to us anymore. Still, we just select the columns we're interested in (i.e. id and 4 columns per label - 3 labels per call was the maximum). We then need to piece the information from the separate columns together to adhere to the label representation used in MLStudio. We drop the calls that weren't labeled too.

# In[ ]:


df_excel = pd.read_excel(
    INPUT_EXCEL,
    sheet_name="DATA",
    usecols=[
        "id",
        "contact_type_1",
        "grouping_1",
        "specification_1_0",
        "specification_1_1",
        "contact_type_2",
        "grouping_2",
        "specification_2_0",
        "specification_2_1",
        "contact_type_3",
        "grouping_3",
        "specification_3_0",
        "specification_3_1",
    ],
)

df_excel.dropna(subset=["contact_type_1"], inplace=True)


def combine_columns(row, columns):
    return "/".join([str(row[col]) for col in columns if row[col] is not None])


df_excel["label_1"] = df_excel.apply(
    lambda row: combine_columns(
        row, ["contact_type_1", "grouping_1", "specification_1_0", "specification_1_1"]
    ),
    axis=1,
)
df_excel["label_2"] = df_excel.apply(
    lambda row: combine_columns(
        row, ["contact_type_2", "grouping_2", "specification_2_0", "specification_2_1"]
    ),
    axis=1,
)
df_excel["label_3"] = df_excel.apply(
    lambda row: combine_columns(
        row, ["contact_type_3", "grouping_3", "specification_3_0", "specification_3_1"]
    ),
    axis=1,
)

df_excel["label"] = df_excel.apply(
    lambda row: [
        row["label_1"].replace("/nan", ""),
        row["label_2"].replace("/nan", ""),
        row["label_3"].replace("/nan", ""),
    ],
    axis=1,
)
df_excel["label"] = df_excel["label"].apply(
    lambda x: [label for label in x if label != "nan"]
)
df_excel["label"] = df_excel["label"].apply(lambda x: ",".join(x))

df_excel = df_excel[["id", "label"]]

df_excel.head()


# In[ ]:


print(f"Excel data shape: {df_excel.shape}")


# #### Label check
# 
# Let's check whether we only have labels left that we expect.

# In[ ]:


df_excel["label"] = df_excel["label"].apply(
    lambda x: [x for x in x.split(",")]
)
df_excel = df_excel.explode("label")

print(len(df_excel["label"].unique()))

# Uncomment below if you want to see the labels and how often they occur.
# Note that these numbers differ from the distributions later, since this takes all labels per call into account.
# Later on, we only take the first label into account.
# df_excel.label.value_counts(normalize=True)


# In[ ]:


# 'Implode' the data again, so that we have one row per call.
df_excel = df_excel.groupby("id")["label"].apply(lambda x: ",".join(x)).reset_index()
df_excel.sort_values("id", inplace=True)
df_excel.head()


# ## Merge Data
# 
# Obviously we have to paste these two dataframes together now to create one big set.

# In[ ]:


df = pd.concat([df_mlstudio, df_excel], ignore_index=True)
df.info()


# ## Create Test Set
# 
# We'll split the dataset in two now. The majority of it will be used to train models. The smaller set will be used afterwards to assess model performance. We'll use a stratified split (i.e. approximately the same percentages per label in both sets). Since there are a whole lot of potential label combinations for a multilabel problem, we will do this based on the first label for each call.

# In[ ]:


df["label_1"] = df["label"].apply(lambda x: x.split(",")[0] if len(x.split(",")) > 0 else "")

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_1"]
)


# In[ ]:


df_train.label_1.value_counts(normalize=True)


# In[ ]:


df_test.label_1.value_counts(normalize=True)


# In[ ]:


df_train.drop(columns=["label_1"], inplace=True)
df_test.drop(columns=["label_1"], inplace=True)


# ## Store Data
# 
# As in all these notebooks, we'll store this intermediate step so you do not necessarily have to run all of them in order.

# In[ ]:


df_train.to_csv(f"{PATH_LABELED}/labeled_train_{DEPARTMENT}.csv", index=False)
df_test.to_csv(f"{PATH_LABELED}/labeled_test_{DEPARTMENT}.csv", index=False)

