# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch


# Importing the dataset
dataset = pd.read_csv('final_dataset.csv')
data_list = dataset.values.tolist()

X = dataset.iloc[:, [2, 0]].values  # 0 Rooms, 1 Area, 2 Price, 3 Rating

print(X)

# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))
plt.title('Дендрограма complete link')
plt.xlabel('Кімнат')
plt.ylabel('Евклідові відстані complete link (ціна)')
plt.savefig('plot_images/dendrogram_complete.jpg')
plt.show()

# Training the Hierarchical Clustering model on the dataset
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete')
y_hc = hc.fit_predict(X)

print(y_hc)
print(type(y_hc))

res = []
for i in range(len(y_hc)):
    flat_dict = {'flat': data_list[i],
                 'cluster': y_hc[i]}
    res.append(flat_dict)

print(res)

# Visualising the clusters
colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
for i in range(4):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=f'{colours[i]}',     label=f'Cluster {i}')

plt.title('Кластери квартир')
plt.xlabel('Ціна')
plt.ylabel('Кімнат')
plt.legend()
plt.savefig('plot_images/complete.jpg')
plt.show()