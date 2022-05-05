# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Importing the dataset
dataset = pd.read_csv('final_dataset.csv')
print(dataset)
data_list = dataset.values.tolist()
# print(len(data_list))
X = dataset.iloc[:, [2, 1]].values  # 0 Rooms, 1 Area, 2 Price, 3 Rating

print(X)

# Using the elbow method to find the optimal number of clusters
wcss = []  # within-cluster sum of squares (сума квадратів відстаней від точки до центроїди. більше - гріше)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('"Метод ліктя"')
plt.xlabel('Кількість кластерів')
plt.ylabel('Сума квадратів відстаней від точки до центроїди')
plt.savefig('plot_images/elbow_method.jpg')
plt.show()
#
# # Training the K-Means model on the dataset
# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
# y_kmeans = kmeans.fit_predict(X)
#
# print(y_kmeans)
# # print(len(y_kmeans))
# res = []
# for i in range(len(y_kmeans)):
#     flat_dict = {'flat': data_list[i],
#                  'cluster': y_kmeans[i]}
#     res.append(flat_dict)
#
# print(res)
#
#
# # Visualising the clusters
# colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
# for i in range(4):
#     plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=f'{colours[i]}',     label=f'Cluster {i}')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, c='black', label='Centroids')
# plt.title('Кластери квартир')
# plt.xlabel('Ціна')
# plt.ylabel('Кімнат')
# plt.legend()
# plt.savefig('plot_images/kmeans.jpg')
# plt.show()
