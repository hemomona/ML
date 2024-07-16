if classification can be used, clustering is no need.

# K-MEANS:

pros:

- simple

cons:

- hard to determine K
- complexity is linear with sample size
- can not find cluster with arbitrary shape

## K-MEANS & KNN

K-Means is unsupervised learning for clustering;
KNN is supervised learning for classification.

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

Eps: usually choose k = 2 * num_features - 1, then draw k-distance diagram, and the distance at inflection point is Eps
MinPts: usually k+1

definition:

- core points
- directly density-reachable
- density-reachable
- density-connected
- cluster
- noise

pros:

- no need to determine cluster number
- can find cluster with arbitrary shape
- can find outliers
- just 2 parameters

cons:

- hard for data with many dimensions (features)
- still, hard to determine parameters
- low efficiency

