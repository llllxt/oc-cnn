import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
plt.switch_backend('agg')

test = np.load('feature1.npy').squeeze()
y = np.load('label1.npy').squeeze()
# test = np.load("feature2.npy").squeeze()
# y = np.load('label2.npy').squeeze()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(test)
print("original data dimension is {}. embeded data dimension is {}".format(test.shape[-1], test.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  
plt.figure(figsize=(8, 8))
print(X_norm.shape)
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.savefig('vis.png')
