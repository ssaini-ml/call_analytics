#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


import warnings

import pandas as pd
from hiclass import FlatClassifier
from sklearn.svm import SVC

from call_analytics.hierarchical_evaluation import evaluation_report
from call_analytics.settings import PATH_LABELED, PATH_OPENAI_EMBEDDINGS, RANDOM_STATE

warnings.filterwarnings("ignore")


# # 7_2_1_2 - Classification - Multi-class - Hierarchical - Alternative Hierarchy

# In[ ]:


DEPARTMENT = "PS" # "CS" or "PS"
FEATURE = "summary_customer_only" # "transcription", "summary", "topics", "transcription_customer_only", "summary_customer_only"

if DEPARTMENT == "CS":
    FILE_PREFIX = "20250113_20250212_CS_"
elif DEPARTMENT == "PS":
    FILE_PREFIX = "20250101_20250224_PS_"


# ## Load Data

# In[ ]:


df_labeled_train = pd.read_csv(f"{PATH_LABELED}/labeled_train_{DEPARTMENT}.csv")
df_labeled_test = pd.read_csv(f"{PATH_LABELED}/labeled_test_{DEPARTMENT}.csv")
if FEATURE in ["transcription", "summary", "topics"]:
    df_embeddings = pd.read_csv(f"{PATH_OPENAI_EMBEDDINGS}/{FILE_PREFIX}{FEATURE}_embeddings_text-embedding-3-large.csv")
else:
    df_embeddings = pd.read_csv(f"{PATH_OPENAI_EMBEDDINGS}/{FILE_PREFIX}addedfeatures_{FEATURE}_embeddings_text-embedding-3-large.csv")


# For this experiment, we will only keep the Calls that got exactly one label during labeling.

# In[ ]:


print(f"Number of rows in embeddings: {df_labeled_train.shape[0]}")

# Since something could have gone wrong in preprocessing, we will only keep the embeddings that are in the labeled data
df_labeled_train = df_labeled_train.merge(df_embeddings, on="id", how="inner")
print(f"Number of rows in embeddings after dropping unlabeled IDs: {df_labeled_train.shape[0]}")

# We need to remove all calls with more than one label as well
df_labeled_train = df_labeled_train[df_labeled_train["label"].apply(lambda x: len(x.split(",")) == 1)]
print(f"Number of rows in embeddings after dropping calls with more than one label: {df_labeled_train.shape[0]}")


# In[ ]:


print(f"Number of rows in embeddings: {df_labeled_test.shape[0]}")

# Since something could have gone wrong in preprocessing, we will only keep the embeddings that are in the labeled data
df_labeled_test = df_labeled_test.merge(df_embeddings, on="id", how="inner")
print(f"Number of rows in embeddings after dropping unlabeled IDs: {df_labeled_test.shape[0]}")

# We need to remove all calls with more than one label as well
df_labeled_test = df_labeled_test[df_labeled_test["label"].apply(lambda x: len(x.split(",")) == 1)]
print(f"Number of rows in embeddings after dropping calls with more than one label: {df_labeled_test.shape[0]}")


# ## Hierarchy Definition
# 
# In order to make use of the HiClass library, we need to rework our labels a little bit. We need them to be lists with a value for each level in the hierarchy. Moreover, since we have some overlap in our hierarchy (e.g. Order & Delivery below Support, Informational & Transactional), we will make sure to name every level by the full path towards it.
# 
# ![image](../data/images/20250321%20-%20Hierarchies-CS%20-%20Hierarchy.drawio.png)

# Exactly because of the overlap in level two categories above, we will also try to simplify the hierarchy a bit for the model. We will make it look like the image below, in which we get rid of the first level. Because the bottom-level labels are mostly unique - only the /Other ones are not - we can afterwards still reconstruct most of the original hierarchy.
# 
# ![image](../data/images/20250321%20-%20Hierarchies-CS%20-%20Hierarchy%20-%20Alternative.drawio.png)

# In[ ]:


def refactor_labels_to_alternative(labels: str):
    if labels == "Other":
        return [labels]
    
    temp_labels = labels.split("/")[1:]
    labels = ["/".join(temp_labels[:i]) for i in range(1, len(temp_labels) + 1)]
    
    return labels

df_labeled_train["label"] = df_labeled_train["label"].apply(refactor_labels_to_alternative)
df_labeled_test["label"] = df_labeled_test["label"].apply(refactor_labels_to_alternative)


# ## Modeling
# 
# A hierarchical classifier is a supervised machine learning model in which the output labels are organized according to a predefined hierarchical taxonomy. This setup is useful when labels naturally fall into broader and narrower categories — for example, in customer intent classification where general themes can contain more specific subtopics.
# 
# There are several common strategies for hierarchical classification:
# - Flat Classifier: A standard multi-class classifier that treats all leaf-level labels as independent classes, ignoring the hierarchical relationships between them. This approach serves as a baseline and is useful for comparing performance without added structural complexity.
# - Classifier per Node (Local Classifier per Node – LCN): A binary classifier is trained for each node in the hierarchy (excluding the root). At inference time, classification proceeds top-down through the hierarchy, activating child classifiers only if their parent node is predicted as relevant.
# - Classifier per Parent Node (Local Classifier per Parent Node – LCPN): A multi-class classifier is trained for each parent node to select one of its child nodes. This also follows a top-down traversal and is more efficient than training a binary classifier per node.
# - Classifier per Level (Local Classifier per Level – LCL): A multi-class classifier is trained for each level of the hierarchy. However, this method is not suitable for our use case due to overlap in level 2 labels, where some classes may belong to different parents or appear at the same depth with different semantics.
# 
# ### Approach Used in This Notebook
# In this notebook, we will be using the following classification strategy:
# - A flat multi-class classifier as a baseline, ignoring hierarchical structure.

# In[ ]:


X_train = df_labeled_train.drop(columns=["id", "usage", "label"])
y_train = df_labeled_train["label"]

X_test = df_labeled_test.drop(columns=["id", "usage", "label"])
y_test = df_labeled_test["label"]


# In[ ]:


svc = SVC(probability=True, C=1e-1, kernel="linear", random_state=RANDOM_STATE)

model = FlatClassifier(local_classifier=svc)


# In[ ]:


model.fit(X_train, y_train)


# ### Predictions

# In[ ]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ### Evaluation

# #### Train

# In[ ]:


evaluation_report(y_train, y_train_pred, show_confusion_matrices=False)


# #### Test

# In[ ]:


evaluation_report(y_test, y_test_pred, show_confusion_matrices=False)

