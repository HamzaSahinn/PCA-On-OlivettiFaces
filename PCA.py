# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:41:46 2021

@author: Abdullah Hamza Şahin
"""

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

data = datasets.fetch_olivetti_faces()

X = data.data
y = data.target

pca=PCA()
pca.fit(X)

centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
eigvals, eigvecs = np.linalg.eig(np.cov(X))
plt.plot(eigvals, np.zeros_like(eigvals), 'x')
plt.yticks(())
plt.title("Eigenvalues")
plt.savefig("./output/Eigenvalues.png")
plt.close()

plt.imsave("./output/avarage_image.png",pca.mean_.reshape((64,64)),cmap="gray")


f, axarr = plt.subplots(2,4, figsize=(24,12))



pca=PCA(n_components=5)
pca.fit(X)
X_pca=pca.transform(X)
x_approx = pca.inverse_transform(X_pca)

axarr[0,0].imshow(x_approx[12].reshape((64,64)), cmap='gray')
axarr[0,0].title.set_text("With 5 components")

pca=PCA(n_components=10)
pca.fit(X)
X_pca=pca.transform(X)
x_approx = pca.inverse_transform(X_pca)

axarr[0,1].imshow(x_approx[12].reshape((64,64)), cmap='gray')
axarr[0,1].title.set_text("With 10 components")

pca=PCA(n_components=40)
pca.fit(X)
X_pca=pca.transform(X)
x_approx = pca.inverse_transform(X_pca)

axarr[0,2].imshow(x_approx[12].reshape((64,64)), cmap='gray')
axarr[0,2].title.set_text("With 40 components")

pca=PCA(n_components=200)
pca.fit(X)
X_pca=pca.transform(X)
x_approx = pca.inverse_transform(X_pca)

axarr[0,3].imshow(x_approx[12].reshape((64,64)), cmap='gray')
axarr[0,3].title.set_text("With 200 components")

axarr[1,0].imshow(X[12].reshape((64,64)), cmap="gray")
axarr[1,0].title.set_text("Original image")
axarr[1,1].imshow(X[12].reshape((64,64)), cmap="gray")
axarr[1,1].title.set_text("Original image")
axarr[1,2].imshow(X[12].reshape((64,64)), cmap="gray")
axarr[1,2].title.set_text("Original image")
axarr[1,3].imshow(X[12].reshape((64,64)), cmap="gray")
axarr[1,3].title.set_text("Original image")

f.savefig("./output/Image_Different_Components.png")


pca=PCA(n_components=3)
pca.fit(X)
X_pca=pca.transform(X)


fig=plt.figure(figsize=(7,7))
ax=fig.add_subplot(111, projection='3d')
scatter=ax.scatter(X_pca[:400,0], X_pca[:400,1], X_pca[:400,2],c=y[:400],cmap=plt.get_cmap('jet', 40))
fig.colorbar(scatter)
fig.savefig("./output/ImagesIn_3D.png")