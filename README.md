# PCA-On-OlivettiFaces
This repo contains PCA applications on Olivetti face dataset.


The program first loads the data. The data consist of 400 images and these images belong to 10 different humans. Each image is a 64x64 grayscale. When we vectorized the data it becomes 400x4096. Each row represents one image and images have 4096 features for each. Then the program fits the data to the PCA model. It creates 400 components, then plots the mean image and eigenvalues. After that, it fits different PCA with a different number of components (5,10,40,200). Then it retransforms the 12. image by using these components and the model. As expected, If we decrease the component number we will get more “bad” images compared to the original image.

You can find all of the outputs in output file.
