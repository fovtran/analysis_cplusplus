import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

data=np.array([   5,    1,  100,  102,    3,    4,  999, 1001,    5,    1,    2,    150,  180,  175,  898, 1012], dtype=np.float64)

#data = np.array(normalized)
centroid,_ = kmeans(data, 3)
idx,_ = vq(data, centroid)
X=data.reshape(len(data),1)
Y=centroid.reshape(len(centroid),1)
D_k = cdist( X, Y, metric='euclidean' )
colors = ['red', 'green', 'blue']
pId=range(0,(len(data)-1))
cIdx = [np.argmin(D) for D in D_k]
dist = [np.min(D) for D in D_k]
r=np.vstack((data,dist)).T
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mark=['^','o','>']
for i, ((x,y), kls) in enumerate(zip(r, cIdx)):
    ax.plot(r[i,0],r[i,1],color=colors[kls],marker=mark[kls])
    ax.annotate(str(i), xy=(x,y), xytext=(0.5,0.5), textcoords='offset points',
                 size=8,color=colors[kls])


ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Data')
ax.set_ylabel('Distance')
plt.show()
