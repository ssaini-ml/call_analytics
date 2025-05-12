#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('cd', '..')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from call_analytics.openai_utility import authenticate_openai_client
from call_analytics.settings import PATH_AIRCALL_PROCESSED, PATH_OPENAI_EMBEDDINGS, RANDOM_STATE


# # 4 - CLUSTERING - OpenAI embeddings
# 
# This notebook explores document clustering using OpenAI's `text-embedding-3-large` model. By converting text data into high-dimensional embeddings, we apply K-Means clustering to uncover meaningful groupings within the dataset.
# 
# Once the clusters are formed, we analyze their characteristics by selecting representative examples from each cluster. To enhance interpretability, we use OpenAI's chat completion API to generate descriptive summaries for each cluster, providing insights into the themes present in the dataset.
# 
# ### Objectives
# - Apply K-Means clustering: Identify meaningful groups within the data based on similarity.
# - Generate theme descriptions: Use OpenAI's chat completion API to create human-readable summaries for each cluster.

# ## User Input
# *Note: You will have to choose to number of clusters you want to create in the section [Clustering](#clustering).*

# In[ ]:


DEPARTMENT = "CS" # "CS": Customer Service or "PS": Pharma Service
FEATURE = "summary" # "summary" or "transcription"
DIM_REDUCER = "UMAP" # "t-SNE" or "UMAP"

if DEPARTMENT == "CS":
    INPUT_EMBEDDINGS = f"{PATH_OPENAI_EMBEDDINGS}/20250113_20250212_CS_{FEATURE}_embeddings_text-embedding-3-large.csv"
    INPUT_TEXT = f"{PATH_AIRCALL_PROCESSED}/20250113_20250212_CS.csv"
elif DEPARTMENT == "PS":
    INPUT_EMBEDDINGS = f"{PATH_OPENAI_EMBEDDINGS}/20250101_20250224_PS_{FEATURE}_embeddings_text-embedding-3-large.csv"
    INPUT_TEXT = f"{PATH_AIRCALL_PROCESSED}/20250101_20250224_PS.csv"


# Here we read both the text of interest and the embeddings from file. These were stored to limit the number of calls to Azure OpenAI.

# In[ ]:


text_df = pd.read_csv(INPUT_TEXT, encoding="latin-1", usecols=["id", FEATURE])
embeddings_df = pd.read_csv(INPUT_EMBEDDINGS, encoding="utf-8")
embeddings_df.drop(columns=["usage"], inplace=True)

df = pd.merge(text_df, embeddings_df, on="id", how="inner")
print(f"Calls with summary embeddings: {df.shape[0]}")


# We drop everything here, except for the actual embedding values.

# In[ ]:


embedding_matrix = df.drop(columns=["id", FEATURE])
embedding_matrix = embedding_matrix.to_numpy()
embedding_matrix.shape


# ## Clustering
# 
# The actual clustering is done here. Specify the number of clusters you want as a result below.

# In[ ]:


n_clusters = 6 # Specify the number of clusters you are interested in here

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE)
kmeans.fit(embedding_matrix)
labels = kmeans.labels_
df["cluster"] = labels

df.cluster.value_counts(normalize=True)


# Let's visualize the results of this clustering. It could show something interesting. This visualization will be done in 2D, since we can't comprehend a whole lot more than that. To make that possible, we need to reduce the feature dimensions from 3072 to 2. This is done here either by using UMAP or t-SNE.

# In[ ]:


if DIM_REDUCER == "UMAP":
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    vis_dims2 = umap.fit_transform(embedding_matrix)
elif DIM_REDUCER == "t-SNE":
    tsne = TSNE(
        n_components=2,
        perplexity=15,
        random_state=RANDOM_STATE,
        init="random",
        learning_rate=200,
    )
    vis_dims2 = tsne.fit_transform(embedding_matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

# List of 25 colors, generated using https://medialab.github.io/iwanthue/. If you need more, generate more.
colors = [
    "#c691cc",
    "#73d446",
    "#6a41c8",
    "#d1de4f",
    "#c44fc9",
    "#6fce7e",
    "#c8488b",
    "#7bdcc5",
    "#d24426",
    "#667acf",
    "#d9ad3b",
    "#4a2a6c",
    "#8b963a",
    "#c64152",
    "#6cb3d2",
    "#cf7d39",
    "#5b6380",
    "#cfc890",
    "#642739",
    "#43672d",
    "#d18984",
    "#31322c",
    "#cbc5c9",
    "#7f5432",
    "#5d8975",
]

plt.figure(figsize=(12, 12))
for category, color in enumerate(colors[:n_clusters]):
    xs = np.array(x)[df.cluster == category]
    ys = np.array(y)[df.cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title(f"Clusters identified visualized in 2D using {DIM_REDUCER}")


# ## Naming the clusters
# 
# We're taking some random samples from each cluster here. We then use `gpt-4o` to name the clusters based on those examples.

# In[ ]:


client = authenticate_openai_client()

# Reading a review which belong to each group.
summary_per_cluster = 5

for i in range(n_clusters):
    print(f"Cluster {i}: {df.cluster.value_counts(normalize=True).get(i):.2%} of calls\nTheme:", end=" ")

    reviews = "\n".join(
        df[df.cluster == i]
        .summary.sample(summary_per_cluster, random_state=RANDOM_STATE, replace=True)
        .values
    )

    messages = [
        {
            "role": "user",
            "content": f'What do these pieces of text describing customer service interactions have in common content-wise? Describe the theme in maximally 50 words. Focus on the essentials.\n\nTexts:\n"""\n{reviews}\n"""\n\nTheme:',
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(f"{response.choices[0].message.content.replace('\n', '')}\n")

    sample_cluster_rows = df[df.cluster == i].sample(
        summary_per_cluster, random_state=RANDOM_STATE, replace=True
    )
    for j in range(summary_per_cluster):
        print(sample_cluster_rows[FEATURE].values[j])

    print("-" * 100)

