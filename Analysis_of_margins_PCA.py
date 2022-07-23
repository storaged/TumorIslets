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

import pandas as pd
import glob
import os

path = "/Users/krzysiek/PROJEKTY_NAUKOWE/IMMUCAN/rudy_results/invasive_margins/"

all_files = glob.glob(os.path.join(path, "*50.csv"))

df = pd.concat((pd.read_csv(f).assign(dataset_name=idx) for idx, f in enumerate(all_files)), ignore_index=False)
df = df.fillna(0)

## PCA
data = df.loc[:, df.columns != 'dataset_name']
#x = StandardScaler().fit_transform(data)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data ** 0.5)

print(pca.components_)
print(pca.explained_variance_ratio_)

principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])
target = df[['dataset_name']]
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

colors = ['r', 'g', 'b', 'yellow', 'purple']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['dataset_name'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()
