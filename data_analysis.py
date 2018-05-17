# Import Liabraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Read data
file_path = 'Assignment/foo.csv'
df = pd.read_csv(file_path, names=['data'])

# Basic data exploration
df.describe()
plt.scatter(df.index, df['data'])
sns.distplot(df['data'], bins=50, kde=True)

# From above plot, data seems to have bimodal distribution
# Exploring binary classification
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(df['data'].reshape(len(df['data']),1))
print(kmeans.cluster_centers_) # Check cluster centers --> match the peaks in distribution

# Classified data
plt.scatter(df.index, df['data'], c=kmeans.labels_, cmap='viridis')
plt.plot(df.index, kmeans.cluster_centers_[0]*np.ones(len(df)), color='blue')
plt.plot(df.index, kmeans.cluster_centers_[1]*np.ones(len(df)), color='red')
plt.ylim(df['data'].min()*1.5, df['data'].max()*1.5)
plt.show()

# Let us look at indicidual clusters
df['label'] = kmeans.labels_
df_cluster0 = df[df['label']==0]
df_cluster1 = df[df['label']==1]

df_cluster0.describe()
df_cluster1.describe()

sns.kdeplot(df_cluster0.index, df_cluster0['data'], cmap="Reds", shade=True, shade_lowest=False)
sns.kdeplot(df_cluster1.index, df_cluster1['data'], cmap="Blues", shade=True, shade_lowest=False)

#conclusion - The given data seems to be a bivariate distribution, with cluster centres at 20.25 and -0.03.
#The clusterring shows almost equal population amongst the 2 classes,  Also, the 25%,50% and standard deviation of the
#dataasets show quite similarity,
plt.scatter(df_cluster0.index, df_cluster0.data - df_cluster0.data.mean(),c ='b')
plt.scatter(df_cluster1.index, df_cluster1.data - df_cluster1.data.mean(), c = 'r')

sns.distplot(df_cluster0['data'], bins=50, kde=True, color = 'blue')
sns.distplot(df_cluster1['data'] , bins = 50, kde = True, color= 'red')

# This looks like the same distribution shifted to different mean and mirrored
# So, let's map x1-mean and mean-x2 [x1 being one cluster and x2 being another]
sns.distplot(df_cluster0['data']-df_cluster0.data.mean(), bins=50, kde=True, color = 'blue')
sns.distplot(df_cluster1.data.mean()-df_cluster1['data'] , bins = 50, kde = True, color= 'red')
#  As expected the distributions match a lot!!
