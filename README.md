<div align="center">
  <img src="https://raw.githubusercontent.com/Chrysippean/Music-Clustering-and-Classification/master/musicclassifier.png" width="500" />
</div>

<h1 align="center">
  Music Clustering & Classification: Machine Learning with Matlab
</h1>

## Summary

The goal of this project is to explore the the process in which machine learning could be used to recognize & classify music according to genre, and simultaneously compare the effectiveness of different supervised & unsupervised ML algorithms to this end. The algorithms tested include:

1. Naïve Bayes
2. Linear Discriminant Analysis
3. K-Nearest Neighbors

The sound dataset used included 30 second samples of tracks from 10 different genres, such as blues, classical, and hiphop. For the sake of computation speed, these tracks were trimmed down to 5 seconds. In order to distinguish the genres, spectrograms for each genre was generated in order to obtain sound frequency, which was then used as input for Singular Value Decomposition (SVD). The resulting modes served as the defining attribute for each genre. The songs are then randomized before applying any of the above listed algorithms.

The 'report.docx' file contains a more detailed layout regarding the mathematics, procedure, and the results of this project. 

## NOTE:
Due to the large size of the music folder used in this project, it is not included in the repository. If you want to download the files, you can find them at [http://www.marsyas.info/sound/genres/].
