# Machine-Learning-Nutrition-Diary
## Used Model
The K-means algorithm from the Clustering method, which is an unsupervised learning method, was used in the project.
### K-Means Algorithm
K-Means algorithm is an unsupervised learning and clustering algorithm. The K value in K-Means determines the number of clusters and it must take this value as a parameter. It is called K-Means because it creates K unique clusters and the center of each cluster is the average of the values ​​in the cluster.
The two most important goals here are:
• The values ​​within the cluster should be most similar to each other,
• Clusters should not be as similar to each other as possible

## Algorithm Steps
1. Initial phase:
K random centroid points are selected. K is a predetermined parameter and determines the number of clusters.

2. Assignment phase:
      Each data point is associated with a cluster based on the nearest centroid.
      Euclidean distance is generally used as a distance measurement.
                  Each data point is assigned to the nearest centroid and included in that centroid's cluster.

3. Center recalculation phase:
The center of each cluster is recalculated. New centers are determined as the average of the data points within the cluster. That is, the centroid for each cluster is calculated as the average of the coordinates of the data points within the cluster.

4. Iteration phase:
The assignment and center recalculation steps are repeated until the centers and cluster assignments remain unchanged.
Iterations continue until the centers are fixed.

5. Conclusion phase:
The K-means algorithm completes the clustering process as a result of iterations.
Each data point is assigned to the closest of the final centroids and is considered part of that cluster.
The result obtained is a clustering result in which data points are grouped into a certain number of clusters.
## Elbow Method
Elbow Method is a method used to determine the correct number of clusters in the K-means algorithm. This method is based on applying the K-means algorithm for different numbers of clusters and calculating inertia values ​​for each number of clusters.

The inertia value represents the sum of the squared distances of the data points within a cluster from the center.

1. To begin with, we will apply the K-means algorithm for different numbers of clusters (K values).

2. When the K-means algorithm is applied for each number of clusters, the distance of each data point to the cluster center is calculated and the inertia value is obtained by adding the squares of these distances.

3. After the inertia values ​​are calculated for different cluster numbers, the graph of these values ​​is drawn. In the graph, the number of clusters (K value) is positioned on the x-axis and the inertia value is positioned on the y-axis.

4. Observe on the graph how inertia values ​​change depending on the number of clusters. The inertia value generally decreases as the number of clusters increases, because more clusters can have closer centers.

5. The point on the graph where the rate of decrease of inertia values ​​slows down is called the "elbow" point. This point represents the point at which inertia values ​​slow down from decreasing rapidly and the impact of the number of extra clusters on performance is limited.


6. According to the Elbow Method, the elbow point is generally considered the correct number of clusters. This point represents the number of clusters that achieve optimal balance.

## Elbow Graphic
![image](https://github.com/SenaAydin7/Machine-Learning-Nutrition-Diary/assets/92725053/daedb0f4-56ca-48e1-b6e8-3d5f5ebaf0c4)

## The two clusters: Good/Bad nutrients and their centers
![image](https://github.com/SenaAydin7/Machine-Learning-Nutrition-Diary/assets/92725053/d0cdcb33-66c8-48c6-a8ce-70e2531be0fc)
