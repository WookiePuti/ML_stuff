import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection


df = pd.read_excel('titanic.xls')
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

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(y))
