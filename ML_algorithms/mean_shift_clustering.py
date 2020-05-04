import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection


df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

df.fillna(0, inplace=True)
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
df.drop(['name', 'body', 'ticket', 'boat'], 1, inplace=True)
# df['ticket'] = df['ticket'].str.extract('(\d+)', expand=False)
# df['ticket'].convert_dtypes()

df = pd.get_dummies(df, drop_first=True)
# print(df.columns)
# print(df.dtypes)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i, label in enumerate(labels):
    original_df['cluster_group'].iloc[i] = label

n_clusters = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
