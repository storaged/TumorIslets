import tarfile
import urllib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import pandas as pd
import glob
import os

path = "/Users/krzysiek/PROJEKTY_NAUKOWE/IMMUCAN/invasive_margins_maxdist_30_alpha_30/"

all_files = glob.glob(os.path.join(path, "*30.csv"))

df = pd.concat((pd.read_csv(f).assign(dataset_name=idx) for idx, f in enumerate(all_files)), ignore_index=False)
df = df.fillna(0)


res = df.loc[:, df.columns.isin(['Margin Number', 'dataset_name'])]
df = df.loc[:, ~df.columns.isin(['Margin Number', 'dataset_name'])]

x = df.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)
df = pd.DataFrame( (x/x.sum(axis=1, keepdims=True))**0.5 )

print(df.head(100))
label_encoder = LabelEncoder()

true_labels = label_encoder.fit_transform(res['dataset_name'].tolist())
n_clusters = len(label_encoder.classes_)

preprocessor = Pipeline(
    [
        #("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)
clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=50,
                max_iter=500,
                random_state=42,
            ),
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

data = df  # .loc[:, df.columns != 'dataset_name']
pipe.fit(data)
preprocessed_data = pipe["preprocessor"].transform(data)
predicted_labels = pipe["clusterer"]["kmeans"].labels_
silhouette_score(preprocessed_data, predicted_labels)
adjusted_rand_score(true_labels, predicted_labels)

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

print(pipe["clusterer"]["kmeans"].cluster_centers_)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=40,
    data=pcadf,
    hue="true_label",
    style="predicted_cluster",
    palette="Set2",
)

scat.set_title(
    "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()

