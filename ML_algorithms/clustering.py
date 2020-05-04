import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1, 0.6],
              [9,11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k.", "o."]
for index, point in enumerate(X):
    plt.plot(X[index][0], X[index][1], colors[labels[index]], markersize = 10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()



