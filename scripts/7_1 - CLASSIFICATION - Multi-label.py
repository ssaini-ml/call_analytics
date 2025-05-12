#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from call_analytics.settings import PATH_LABELED, PATH_OPENAI_EMBEDDINGS, RANDOM_STATE
from call_analytics.hierarchical_evaluation import evaluation_report


# # 7_3_1 - Classification - Multi-label

# In[ ]:


DEPARTMENT = "CS" # "CS" or "PS"
FEATURE = "topics" # "transcription", "summary", "topics", "transcription_customer_only", "summary_customer_only"

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


# In[ ]:


print(f"Number of rows in embeddings: {df_labeled_train.shape[0]}")

# Since something could have gone wrong in preprocessing, we will only keep the embeddings that are in the labeled data
df_labeled_train = df_labeled_train.merge(df_embeddings, on="id", how="inner")

print(f"Number of rows in embeddings after dropping unlabeled IDs: {df_labeled_train.shape[0]}")


# In[ ]:


print(f"Number of rows in embeddings: {df_labeled_test.shape[0]}")

# Since something could have gone wrong in preprocessing, we will only keep the embeddings that are in the labeled data
df_labeled_test = df_labeled_test.merge(df_embeddings, on="id", how="inner")

print(f"Number of rows in embeddings after dropping unlabeled IDs: {df_labeled_test.shape[0]}")


# ## Modeling
# 
# ### Hierarchy Level 1
# We will focus on this first level of the hierarchy to start with. We're taking it on as a multi-label problem. These are the steps we will take:
# 1. Create targets that include only unique level 1 labels
# 2. Binarize targets
# 3. Define models and parameter grids
# 4. Define k-fold cross validation (can't use stratified approach with multi-label problem)
# 5. Define the scorer
# 6. Store best models
# 7. Evaluate best models

# In[ ]:


# Let's first create a column for level_1 labels only. We keep unique ones only.
df_labeled_train["level_1"] = df_labeled_train["label"].apply(lambda x: list(set([x.split("/")[0] for x in x.split(",")])))
df_labeled_test["level_1"] = df_labeled_test["label"].apply(lambda x: list(set([x.split("/")[0] for x in x.split(",")])))


# In[ ]:


y_train = df_labeled_train["level_1"]
X_train = df_labeled_train.drop(columns=["id", "usage", "label", "level_1"])

y_test = df_labeled_test["level_1"]
X_test = df_labeled_test.drop(columns=["id", "usage", "label", "level_1"])


# In[ ]:


mlb = MultiLabelBinarizer()
y_bin_train = mlb.fit_transform(y_train)
print(f"Level 1 labels for case 0: {df_labeled_train.at[0, 'level_1']}")
print(f"Level 1 labels in MultiLAbelBinarizer: {mlb.classes_}")
print(f"Level 1 binarized labels for case 0: {y_bin_train[0]}")


# In[ ]:


models = {
    'SVC': {
        'model': OneVsRestClassifier(SVC(probability=True)),  # SVC needs probability=True for OVR
        'params': {
            'estimator__C': [0.1, 1],  # Regularization strength
            'estimator__kernel': ['linear'],  # Different decision boundaries
        }
    }
}


# In[ ]:


# Define K-Fold for multilabel classification
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# In[ ]:


# Define scoring function (F1 score is often better for multilabel problems)
scorer = make_scorer(f1_score, average='micro')


# In[ ]:


best_models = {}

# Iterate over models and perform Grid Search
for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")
    
    grid_search = GridSearchCV(
        model_info['model'], 
        model_info['params'], 
        cv=cv, 
        scoring=scorer, 
        n_jobs=-1, 
        verbose=2
    )
    
    grid_search.fit(X_train, y_bin_train)
    
    # Store the best model for each classifier
    best_models[model_name] = grid_search.best_estimator_
    
    # Print the best hyperparameters
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")


# In[ ]:


y_pred_train = best_models['SVC'].predict(X_train)

# for num, label in enumerate(mlb.classes_):
#     cm = multilabel_confusion_matrix(y_bin_test, y_pred)
#     print(f"Label: {label}")
#     print(f"Confusion Matrix: {cm}")
#     print("\n")
print(classification_report(y_bin_train, y_pred_train, target_names=mlb.classes_, zero_division=0))


# In[ ]:


y_bin_test = mlb.transform(y_test)
y_pred_test = best_models['SVC'].predict(X_test)

# for num, label in enumerate(mlb.classes_):
#     cm = multilabel_confusion_matrix(y_bin_test, y_pred)
#     print(f"Label: {label}")
#     print(f"Confusion Matrix: {cm}")
#     print("\n")
print(classification_report(y_bin_test, y_pred_test, target_names=mlb.classes_, zero_division=0))


# ## Full hierarchy - Flat
# 
# Here we'll do flat classification for all labels in the complete hierarchy. We use the same steps as specified above.

# In[ ]:


df_labeled_train["full_label"] = df_labeled_train["label"].apply(lambda x: x.split(","))
df_labeled_test["full_label"] = df_labeled_test["label"].apply(lambda x: x.split(","))


# In[ ]:


y_train = df_labeled_train["full_label"]
X_train = df_labeled_train.drop(columns=["id", "usage", "label", "level_1", "full_label"])

y_test = df_labeled_test["full_label"]
X_test = df_labeled_test.drop(columns=["id", "usage", "label", "level_1", "full_label"])


# In[ ]:


mlb = MultiLabelBinarizer()
mlb.fit(y_train)
y_bin_train = mlb.transform(y_train)
y_bin_test = mlb.transform(y_test)
print(f"Labels for case 0: {df_labeled_train.at[0, 'full_label']}")
print(f"Labels in MultiLAbelBinarizer: {mlb.classes_}")
print(f"Binarized labels for case 0: {y_bin_train[0]}")


# In[ ]:


models = {
    'SVC': {
        'model': OneVsRestClassifier(SVC(probability=True)),  # SVC needs probability=True for OVR
        'params': {
            'estimator__C': [1],  # Regularization strength
            'estimator__kernel': ['rbf'],  # Different decision boundaries
            'estimator__gamma': ['scale'],  # Kernel coefficient
            'estimator__class_weight': ['balanced']   # Class weights
        }
    }
}


# In[ ]:


# Define K-Fold for multilabel classification
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# In[ ]:


# Define scoring function (F1 score is often better for multilabel problems)
scorer = make_scorer(f1_score, average='micro')


# In[ ]:


best_models = {}

# Iterate over models and perform Grid Search
for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")
    
    grid_search = GridSearchCV(
        model_info['model'], 
        model_info['params'], 
        cv=cv, 
        scoring=scorer, 
        n_jobs=-1, 
        verbose=2
    )
    
    grid_search.fit(X_train, y_bin_train)
    
    # Store the best model for each classifier
    best_models[model_name] = grid_search.best_estimator_
    
    # Print the best hyperparameters
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")


# In[ ]:


y_pred_train = best_models['SVC'].predict(X_train)

# for num, label in enumerate(mlb.classes_):
#     cm = multilabel_confusion_matrix(y_bin_test, y_pred)
#     print(f"Label: {label}")
#     print(f"Confusion Matrix: {cm}")
#     print("\n")
print(classification_report(y_bin_train, y_pred_train, target_names=mlb.classes_, zero_division=0))


# In[ ]:


y_pred_test = best_models['SVC'].predict(X_test)

# for num, label in enumerate(mlb.classes_):
#     cm = multilabel_confusion_matrix(y_bin_test, y_pred)
#     print(f"Label: {label}")
#     print(f"Confusion Matrix: {cm}")
#     print("\n")
print(classification_report(y_bin_test, y_pred_test, target_names=mlb.classes_, zero_division=0))

