# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing
from sklearn.cluster import KMeans

plt.style.use('fivethirtyeight')

df = pd.read_csv('dataset.csv')
df = df.drop(labels='ClientId', axis=1)
#выводим первые 10 элементов
df.head(10)

df.describe()

#выводим наши исходные даные и видим, что все сливается
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After MinMaxScaler')
sns.kdeplot(df['AverageSpeed'], ax=ax1)
sns.kdeplot(df['CountN_Speed'], ax=ax1)
sns.kdeplot(df['CountN_PickAcceleration'], ax=ax1)
sns.kdeplot(df['CountN_BackAcceleration'], ax=ax1)
sns.kdeplot(df['AverageBackAcceleration'], ax=ax1)
sns.kdeplot(df['CountN_LeftRightAcceleration'], ax=ax1)
sns.kdeplot(df['AverageLeftRightAcceleration'], ax=ax1);

#нормализуем данные к одному диапазону от 0 до 1 с помощью MinMaxScaler
mm_scaler = preprocessing.MinMaxScaler()
df_mm = mm_scaler.fit_transform(df)
col_names = list(df.columns)
df_mm = pd.DataFrame(df_mm, columns=col_names)

#выводим график с нормализованными данными
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After MinMaxScaler')
sns.kdeplot(df_mm['AverageSpeed'], ax=ax1)
sns.kdeplot(df_mm['CountN_Speed'], ax=ax1)
sns.kdeplot(df_mm['CountN_PickAcceleration'], ax=ax1)
sns.kdeplot(df_mm['CountN_BackAcceleration'], ax=ax1)
sns.kdeplot(df_mm['AverageBackAcceleration'], ax=ax1)
sns.kdeplot(df_mm['CountN_LeftRightAcceleration'], ax=ax1)
sns.kdeplot(df_mm['AverageLeftRightAcceleration'], ax=ax1);

#сравниваем как выглядели отдельные графики до и после нормализации. Это график средней скорости до
fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 4))
ax1.set_title('After MinMaxScaler')
sns.kdeplot(df['AverageSpeed'], ax=ax1)

#Это график средней скорости после нормализации. Вывод - форма не изменилась, т.е. расстояние между данными сохранилось
fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 4))
ax1.set_title('After MinMaxScaler')
sns.kdeplot(df_mm['AverageSpeed'], ax=ax1)

#выводим нормализованные данные (первые 10)
df_mm.head(10)

maxs = [df_mm[col].max() for col in df_mm.columns]
maxs

#для информации о свойствах выводим гистограмму с распределением всех таблиц
df_mm.hist();

#смотрим на корреляцию между свойствами
sns.heatmap(df_mm.corr());

#определяем оптимальное количество кластеров (с нормализованными данными их стало больше)
x = df_mm.iloc[:, 0:10].values
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.show()

#проводим кластеризацию методом k-средних
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'cluster 4')
# plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'cluster 5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

# plt.style.use('fivethirtyeight')
plt.title('K-Means Clustering', fontsize = 20)

plt.legend()
plt.grid()
plt.show()

km.fit_predict(x)

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 30)

plt.show()