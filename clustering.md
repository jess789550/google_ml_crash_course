# Clustering
## What is clustering?
- find patterns and similarities in the dataset
- unsupervised ML
- unlabeled examples
- similarity measure
- each group is assigned a unique label called a cluster ID

## Clustering use cases
- Market segmentation
- Social network analysis
- Search result grouping
- Medical imaging
- Image segmentation
- Anomaly detection

### Other terminology
- Imputation: Missing data can be inferred from other data
- Data compression: The relevant cluster ID can replace other features - reduce number of features and resources needed
- Privacy presentation: Cluster users and associate user data with cluster IDs instead of user IDs

## CLustering algorithms
- k-means algorithm: complexity of O(n) meaning that the algorithm scales with n
- Centroid-based clustering: non-hierachial clusters, sensitive to ouliers, k-means
- Density-based clustering: connects contiguous areas of high example density into clusters
- Distribution-based clustering: probablistic distributions, Gaussian
- Hierachial clustering: tree, taxonomies

![Centroid-based clustering](images/centroid.png)

![Density-based clustering](images/density.png)

![Distribution-based clustering](images/distribution.png)

![Hierachial clustering](images/hierachial.png)

## Clustering workflow
- Prepare data: normalize, scale, and transform feature data
- Create similarity metric
- Run clustering algorithm: k-means
- Interpret results and adjust clustering

## Data preparation
- Normalising data: calculate Z-scores for Gaussian distribution (normal distribution)
- Log transforms for power law distribution (positive skew)
- Quantiles for sparse distribution

![Normalise](NormalizeData.png)

![Log transform](logtransform.png)

![Quantiles](Quantize.png)

---

## K-means clustering

