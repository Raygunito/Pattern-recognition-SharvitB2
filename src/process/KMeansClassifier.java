package process;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;

import data.CharacteristicVector;
import data.Classifier;
import data.MathUtilsException;
import logger.LoggerUtil;
import utils.MathUtils;

public class KMeansClassifier implements Classifier {
    private static final Logger logger = LoggerUtil.getLogger(KMeansClassifier.class, Level.WARN);
    private String distanceMetric;
    private int k;
    private int norm;

    private ArrayList<ArrayList<CharacteristicVector>> cluster;
    private ArrayList<CharacteristicVector> arrayCentroid;
    private ArrayList<CharacteristicVector> trainData;

    public KMeansClassifier(int kCluster, String metric) {
        this.k = kCluster;
        this.distanceMetric = metric;
    }

    public KMeansClassifier(int kCluster, String metric, int norm) {
        if (norm < 1) {
            logger.error("Norm is lower than 1, forcing norm to be equal to 1");
            norm = 1;
        }
        this.k = kCluster;
        this.distanceMetric = metric;
        this.norm = norm;
    }

    /**
     * Trains the classifier on the given training data by partitioning the data
     * into k clusters.
     *
     * @param trainingData the data to train the model on
     */
    @Override
    public void train(List<CharacteristicVector> trainingData) {
        this.trainData = new ArrayList<>(trainingData);
        this.arrayCentroid = initCentroid();
        cluster = resetArrayCluster();
        boolean same = false;
        int iteration = 0;
        while (!same) {
            logger.debug("Iteration {}: Reassigning data points to clusters", iteration);
            cluster = resetArrayCluster();

            // Add cVector to the corresponding i-th cluster who has the lowest distance
            // between cvector and the cluster
            for (CharacteristicVector cVector : trainingData) {
                double[] distanceFromCentroid = arrayDistanceFromCentroid(cVector, arrayCentroid);
                int indexToAdd = minIndex(distanceFromCentroid);
                cluster.get(indexToAdd).add(cVector);
            }

            double currentSSE = calculateSSE();
            logger.info("Iteration {}: SSE = {}", iteration, currentSSE);

            ArrayList<CharacteristicVector> newCentroid = calculateNewCentroids();

            // stop if clusters are stabilized between the previous step and the current
            // step
            if (centroidsAreCloseEnough(newCentroid, arrayCentroid, 0.0000001)) {
                logger.info("Convergence reached after {} iterations", iteration);
                same = true;
            } else {
                logger.debug("Iteration {}: Centroids updated", iteration);
            }
            this.arrayCentroid = newCentroid;
            iteration++;
        }
    }

    // TODO
    @Override
    public String predict(CharacteristicVector vector) throws IllegalStateException {
        if (trainData == null || arrayCentroid == null) {
            logger.error("Attempted to predict without training the model");
            throw new IllegalStateException("Training data not set. Call train() before predict().");
        }

        double[] distances = arrayDistanceFromCentroid(vector, arrayCentroid);
        int nearestCentroidIndex = minIndex(distances);

        String predictedClusterLabel = "Cluster " + nearestCentroidIndex;
        logger.trace("Predicted cluster for input vector: {}", predictedClusterLabel);

        return predictedClusterLabel;
    }

    public ArrayList<ArrayList<CharacteristicVector>> getCluster() {
        return cluster;
    }

    /**
     * Resets the cluster array by creating empty lists for each cluster.
     *
     * @return an ArrayList of empty clusters
     */
    private ArrayList<ArrayList<CharacteristicVector>> resetArrayCluster() {
        ArrayList<ArrayList<CharacteristicVector>> clust = new ArrayList<ArrayList<CharacteristicVector>>();
        for (int i = 0; i < k; i++) {
            clust.add(new ArrayList<CharacteristicVector>());
        }
        logger.trace("Clusters reset to empty");
        return clust;
    }

    /**
     * Using KMeans++ to initialize the centroid
     * <ol>
     * <li>Randomly select the first centroid from the data points.</li>
     * <li>For each data point compute its distance from the nearest, previously
     * chosen centroid.</li>
     * <li>Select the next centroid from the data points such that the probability
     * of choosing a point as centroid is directly proportional to its distance from
     * the nearest, previously chosen centroid. (i.e. the point having maximum
     * distance from the nearest centroid is most likely to be selected next as a
     * centroid)</li>
     * <li>Repeat steps 2 and 3 until k centroids have been sampled</li>
     * </ol>
     * {@link https://www.geeksforgeeks.org/ml-k-means-algorithm/}
     * 
     * @return the arraylist of centroid
     */
    private ArrayList<CharacteristicVector> initCentroid() {
        ArrayList<CharacteristicVector> centroids = new ArrayList<>();
        // randomly take the first centroid
        int sizeData = trainData.size();
        int firstIndex = new Random().nextInt(sizeData);
        centroids.add(trainData.get(firstIndex));
        logger.debug("Selected initial centroid at index {}", firstIndex);

        for (int i = 0; i < this.k - 1; i++) {
            double[] distances = new double[sizeData];
            double sum = 0;

            // Calculate the distance from each point to the nearest centroid
            for (int j = 0; j < sizeData; j++) {
                CharacteristicVector point = trainData.get(j);

                // Find the minimum distance to any centroid in the current centroids list
                double minDist = Double.MAX_VALUE;
                for (CharacteristicVector centroid : centroids) {
                    double dist = calculateDistance(point, centroid);
                    minDist = Math.min(minDist, dist);
                }

                // Store the minimum distance for probability computation
                distances[j] = minDist;
                sum += distances[j];
            }
            // choose the next centroid based on weighted probability
            double cumulativeProbability = 0;
            double threshold = new Random().nextDouble() * sum;

            int nextCentroidIndex = 0;
            int j = 0;
            while (j < sizeData && cumulativeProbability < threshold) {
                cumulativeProbability += distances[j];
                nextCentroidIndex = j;
                j++;
            }

            centroids.add(trainData.get(nextCentroidIndex));
            logger.debug("Added centroid at index {}", nextCentroidIndex);
        }
        logger.info("Initial centroids selected");
        return centroids;
    }

    /**
     * Calculates new centroids as the mean of each cluster's vectors.
     *
     * @return a list of new centroids
     */
    private ArrayList<CharacteristicVector> calculateNewCentroids() {
        ArrayList<CharacteristicVector> nCentroids = new ArrayList<>();
        int vectorSize = this.trainData.get(0).getVectorSize();
        int clusterNumber = 0;
        // Cluster i-th 1->k contains a array of CharactVect we then compute the mean of
        // this array and create a single CVect then add to ncentroid
        for (ArrayList<CharacteristicVector> currentCluster : this.cluster) {
            clusterNumber++;
            // Check if get(0) is not empty !!!
            if (currentCluster.isEmpty()) {
                int randomIndex = new Random().nextInt(trainData.size());
                nCentroids
                        .add(new CharacteristicVector(trainData.get(randomIndex).getVector(),
                                "Cluster " + clusterNumber,
                                "null", "null"));
                logger.warn("Cluster {} was empty; assigned a random data point as centroid", clusterNumber - 1);
                continue;
            }
            double[] sum = new double[vectorSize];
            for (CharacteristicVector cVector : currentCluster) {
                for (int i = 0; i < vectorSize; i++) {
                    sum[i] += cVector.getVector()[i];
                }
            }
            for (int i = 0; i < vectorSize; i++) {
                sum[i] /= currentCluster.size();
            }

            nCentroids.add(new CharacteristicVector(sum, "null", "null", "null"));
        }

        return nCentroids;
    }

    /**
     * Calculates the distance between two vectors based on the selected distance
     * metric.
     *
     * @param v1 the first vector
     * @param v2 the second vector
     * @return the distance between v1 and v2
     */
    private double calculateDistance(CharacteristicVector v1, CharacteristicVector v2) {
        logger.trace("Calculating distance between vectors using metric: {}", distanceMetric);
        try {
            switch (distanceMetric) {
                case Classifier.EUCLIDEAN:
                    return MathUtils.distEuclidean(v1, v2);
                case Classifier.MANHATTAN:
                    return MathUtils.distManhattan(v1, v2);
                case Classifier.MINKOWSKI:
                    System.out.println(norm);
                    return MathUtils.distMinkowski(v1, v2, norm);
                default:
                    logger.warn("Unknown distance metric: {}. Defaulting to maximum distance.", distanceMetric);
                    return Double.MAX_VALUE;
            }
        } catch (MathUtilsException e) {
            logger.error("Error calculating distance between vectors: {}", e.getMessage());
            return Double.MAX_VALUE;
        }
    }

    /**
     * Computes the distances from a vector to each centroid.
     *
     * @param cVector       the vector to compute distances from
     * @param arrayCentroid the list of centroids
     * @return an array of distances from the vector to each centroid
     */
    private double[] arrayDistanceFromCentroid(CharacteristicVector cVector,
            ArrayList<CharacteristicVector> arrayCentroid) {
        double[] distanceCentroid = new double[arrayCentroid.size()];
        for (int i = 0; i < arrayCentroid.size(); i++) {
            distanceCentroid[i] = calculateDistance(cVector, arrayCentroid.get(i));
        }
        return distanceCentroid;
    }

    /**
     * Finds the index of the smallest value in an array.
     *
     * @param array the array to search
     * @return the index of the smallest value
     */
    private int minIndex(double[] array) {
        int i = 0;
        double min = Double.MAX_VALUE;
        for (int j = 0; j < array.length; j++) {
            if (array[j] < min) {
                i = j;
                min = array[j];
            }
        }
        logger.trace("Minimum index found: {}", i);
        return i;
    }

    /**
     * Checks if the centroids have converged within a specified threshold.
     *
     * @param newCentroids the updated centroids
     * @param oldCentroids the previous centroids
     * @param threshold    the maximum allowed difference between corresponding
     *                     centroids
     * @return true if centroids are within the threshold, false otherwise
     */
    private boolean centroidsAreCloseEnough(ArrayList<CharacteristicVector> newCentroids,
            ArrayList<CharacteristicVector> oldCentroids,
            double threshold) {
        for (int i = 0; i < newCentroids.size(); i++) {
            double[] newVector = newCentroids.get(i).getVector();
            double[] oldVector = oldCentroids.get(i).getVector();

            for (int j = 0; j < newVector.length; j++) {
                if (Math.abs(newVector[j] - oldVector[j]) > threshold) {
                    logger.debug("Centroids not close enough at index {}: diff = {}", i,
                            Math.abs(newVector[j] - oldVector[j]));
                    return false; // Centroids are not close enough
                }
            }
        }
        logger.debug("Centroids have converged within the threshold");
        return true;
    }

    /**
     * Calculates the Sum of Squared Errors (SSE) for the current clustering.
     *
     * @return the SSE value
     */
    public double calculateSSE() {
        double sse = 0.0;

        // Loop through each cluster and calculate the squared distances
        for (int i = 0; i < cluster.size(); i++) {
            ArrayList<CharacteristicVector> currentCluster = cluster.get(i);
            CharacteristicVector centroid = arrayCentroid.get(i);

            for (CharacteristicVector cVector : currentCluster) {
                double distance = 0;
                try {
                    // Here we sqrt at after sum
                    distance = MathUtils.distEuclidean(cVector, centroid);
                } catch (MathUtilsException e) {
                    logger.error(e.getMessage());
                }
                sse += Math.pow(distance, 2); // since euclidean distance we sqrt at the end, we x^2 it to respect the
                                              // SSE formula
            }
        }

        return sse;
    }

    /**
     * Calculates the silhouette score for the entire clustering solution.
     * 
     * @return the average silhouette score for the entire clustering solution
     */
    public double calculateSilhouetteScore() {
        ArrayList<ArrayList<CharacteristicVector>> clusters = this.getCluster();
        double totalSilhouetteScore = 0.0;
        int totalPoints = 0;

        // Loop over all data points and calculate the silhouette score for each point
        for (ArrayList<CharacteristicVector> cluster : clusters) {
            for (CharacteristicVector point : cluster) {
                double a = calculateA(point, cluster); // Calculate a(i)
                double b = calculateB(point, clusters); // Calculate b(i)
                totalSilhouetteScore += (b - a) / Math.max(a, b);
                totalPoints++;
            }
        }

        // Return the average silhouette score
        return totalSilhouetteScore / totalPoints;
    }

    /**
     * Calculates the average distance from a data point to all other points in the
     * same cluster (a(i)).
     * 
     * @param point   the data point
     * @param cluster the cluster the point belongs to
     * @return the average distance to all other points in the same cluster
     */
    private double calculateA(CharacteristicVector point, ArrayList<CharacteristicVector> cluster) {
        if (cluster.size() == 1) {
            return 0.0;
        }
        double sum = 0.0;
        for (CharacteristicVector otherPoint : cluster) {
            if (otherPoint != point) {
                try {
                    sum += MathUtils.distEuclidean(point, otherPoint);
                } catch (MathUtilsException e) {
                    // e.printStackTrace();
                }
            }
        }
        return sum / (cluster.size() - 1);
    }

    /**
     * Calculates the average distance from a data point to all points in the
     * nearest cluster (b(i)).
     * 
     * @param point    the data point
     * @param clusters all the clusters
     * @return the average distance to the nearest cluster
     */
    private double calculateB(CharacteristicVector point, ArrayList<ArrayList<CharacteristicVector>> clusters) {
        double minAvgDistance = Double.MAX_VALUE;

        // Find the nearest cluster based on the centroid distance
        for (ArrayList<CharacteristicVector> otherCluster : clusters) {
            if (otherCluster.contains(point))
                continue; // Skip the current cluster

            double sum = 0.0;
            for (CharacteristicVector otherPoint : otherCluster) {
                try {
                    sum += MathUtils.distEuclidean(point, otherPoint);
                } catch (MathUtilsException e) {
                    // e.printStackTrace();
                }
            }
            double avgDistance = sum / otherCluster.size();
            minAvgDistance = Math.min(minAvgDistance, avgDistance);
        }

        return minAvgDistance;
    }
}
