import tarfile
import urllib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.colors as mcolors
import pandas as pd
import glob
import os
import plotly.express as px

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

label_encoder = LabelEncoder()

data = df
#x = StandardScaler().fit_transform(data)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)


principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])
target = res
target.reset_index(inplace=True, drop=True)

finalDf = pd.concat([principalDf,  target], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = finalDf['dataset_name'].unique()

colors = mcolors.TABLEAU_COLORS
#for target, color in zip(targets, colors):
#    indicesToKeep = finalDf['dataset_name'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c=color
#               , s=50)

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    principalComponents,
    labels=labels,
    dimensions=range(4),
    color=colors#df["species"]
)

fig.update_traces(diagonal_visible=True)
fig.show()

ax.legend(targets)
ax.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()
