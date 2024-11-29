package process;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;

import data.CharacteristicVector;
import data.Classifier;
import data.MathUtilsException;
import logger.LoggerUtil;
import utils.MathUtils;

/**
 * Implements the k-Nearest Neighbors (KNN) algorithm
 * for classification. We can uses different distance metrics, including
 * Euclidean, Manhattan, etc. Then we classify input data based on
 * the majority label of its nearest neighbors.
 *
 * The class allows setting the number of neighbors (k) and the distance metric,
 * storing training data without further preprocessing.
 * <p>
 * Example Usage:
 * </p>
 * 
 * <pre>
 * KNNClassifier knn = new KNNClassifier(3, Classifier.EUCLIDEAN);
 * knn.train(trainingData);
 * String label = knn.predict(inputVector);
 * </pre>
 */
public class KNNClassifier implements Classifier {
    private static final Logger logger = LoggerUtil.getLogger(KNNClassifier.class, Level.WARN);

    private String distanceMetric;
    private ArrayList<CharacteristicVector> trainData;
    private int k;
    private int norm;

    /**
     * Constructs a KNNClassifier with a specified number of neighbors and distance
     * metric.
     *
     * @param k          the number of neighbors to consider when classifying an
     *                   input.
     * @param metricName the distance metric to use (e.g., "EUCLIDEAN",
     *                   "MANHATTAN").
     */
    public KNNClassifier(int k, String metricName) {
        logger.info("Initializing KNNClassifier with k={}, metricName={}", k, metricName);
        distanceMetric = metricName;
        this.k = k;
    }

    /**
     * Constructs a KNNClassifier for the Minkowski distance metric with a specified
     * norm.
     * This constructor is specifically for Minkowski distance as it requires an
     * additional parameter.
     *
     * @param k          the number of neighbors to consider when classifying an
     *                   input.
     * @param metricName the distance metric to use, expected to be "MINKOWSKI".
     * @param norm       the norm value for the Minkowski distance metric, must be
     *                   greater than 0.
     */
    public KNNClassifier(int k, String metricName, int norm) {
        if (norm < 1) {
            logger.error("Norm is lower than 1, forcing norm to be equal to 1");
            norm = 1;
        }
        logger.info("Initializing KNNClassifier with k={} and Minkowski norm={}", k, norm);
        distanceMetric = metricName;
        this.k = k;
        this.norm = norm;
    }

    /**
     * Stores the training data for future predictions.
     * <p>
     * In KNN, there is no specific training step, but the training data is stored
     * for
     * use in the prediction phase.
     * </p>
     *
     * @param trainingData a list of {@link CharacteristicVector} representing the
     *                     training dataset.
     */
    @Override
    public void train(List<CharacteristicVector> trainingData) {
        logger.info("Training KNN classifier with {} training samples.", trainingData.size());
        this.trainData = new ArrayList<>(trainingData);
    }

    /**
     * Predicts the label for a given input vector by identifying the k-nearest
     * neighbors from the training data and selecting the majority label among them.
     *
     * Pseudocode:
     * 
     * <pre>
     * function predict(inputVector):
     *     if training data is not set, throw an exception
     *     
     *     distances = empty list to store distances to inputVector
     *     vectors = empty list to store corresponding {@link #CharacteristicVectors}
     *     
     *     for each vector in trainData:
     *         calculate distance between vector and inputVector based on distanceMetric
     *         insert distance and vector in sorted order in distances and vectors
     *
     *     nearest = the first k elements from vectors (k-nearest neighbors)
     *     
     *     return the class label with the highest count from nearest neighbors
     * </pre>
     *
     * @param inputVector the CharacteristicVector representing the data to be
     *                    classified.
     * @return the predicted label based on the majority class among the k-nearest
     *         neighbors.
     * @throws IllegalStateException if training data has not been set.
     * @see CharacteristicVector
     */
    @Override
    public String predict(CharacteristicVector inputVector) throws IllegalStateException {
        logger.debug("Starting prediction for input vector: {}", inputVector);

        if (trainData == null) {
            logger.error("Training data not set. Cannot proceed with prediction.");
            throw new IllegalStateException("Training data not set. Call train() before predict().");
        }

        ArrayList<CharacteristicVector> vectors = this.getNeighbors(inputVector);

        // Grab the k-nearest neighbors
        ArrayList<CharacteristicVector> nearest = new ArrayList<>();
        for (int j = 0; j < k && j < vectors.size(); j++) {
            nearest.add(vectors.get(j));
        }
        logger.debug("Collected and sorted distances for {} neighbors", k);

        // Check which class got the most vote
        String predictedLabel = majorityVote(nearest);
        logger.info("Predicted label: {}", predictedLabel);
        return predictedLabel;
    }

    /**
     * Retrieves all the neighbors of the given input vector in ascending order of
     * distance.
     * If two distances are equal, the vector that was encountered first in the
     * training data will appear before the later one.
     *
     * @param input the CharacteristicVector for which to find the
     *              neighbors.
     *              This vector is compared to all vectors in the training data.
     * @return an ArrayList of CharacteristicVector objects sorted
     *         in ascending order
     *         by their distance to the input vector.
     * @throws IllegalStateException if the training data has not been initialized
     *                               prior to calling this method.
     */
    public ArrayList<CharacteristicVector> getNeighbors(CharacteristicVector input) {
        logger.debug("Retrieving neighbors of : {}", input);

        if (trainData == null) {
            logger.error("Training data not set. Cannot retrieve neighbors.");
            throw new IllegalStateException("Training data not set. Call train() before predict().");
        }
        ArrayList<Double> distances = new ArrayList<Double>();
        ArrayList<CharacteristicVector> vectors = new ArrayList<CharacteristicVector>();

        // Sort each Characteristic Vector by their distance to the InputVector
        for (CharacteristicVector cVector : trainData) {
            int i = 0;
            double distance = calculateDistance(cVector, input);
            while (i < distances.size() && distances.get(i) <= distance) {
                i++;
            }
            distances.add(i, distance);
            vectors.add(i, cVector);
        }
        return vectors;
    }

    /**
     * Retrieves up to k neighbors of the given input vector in ascending order of
     * distance.
     * If two distances are equal, the vector that was encountered first in the
     * training data will appear before the later one.
     * If k is greater than the number of available training data points, all points
     * are returned.
     * 
     * @param input the CharacteristicVector for which to find the neighbors. This
     *              vector is compared to all vectors in the training data.
     * @param k     the maximum number of neighbors to return.
     * @return an ArrayList of CharacteristicVector objects sorted
     *         in ascending order by their distance to the input vector, limited to
     *         k neighbors.
     * @throws IllegalStateException    if the training data has not been
     *                                  initialized prior to calling this method.
     * @throws IllegalArgumentException if k is less than 1.
     */
    public ArrayList<CharacteristicVector> getNeighbors(CharacteristicVector input, int k) {
        logger.debug("Retrieving up to {} neighbors of : {}", k, input);

        if (trainData == null) {
            logger.error("Training data not set. Cannot retrieve neighbors.");
            throw new IllegalStateException("Training data not set. Call train() before predict().");
        }

        if (k < 1) {
            logger.error("Got k={} but getNeighbors should never get an int lower than 1.");
            throw new IllegalArgumentException("The number of neighbors 'k' must be at least 1.");
        }

        ArrayList<Double> distances = new ArrayList<Double>();
        ArrayList<CharacteristicVector> vectors = new ArrayList<CharacteristicVector>();

        // Sort each Characteristic Vector by their distance to the InputVector
        for (CharacteristicVector cVector : trainData) {
            int i = 0;
            double distance = calculateDistance(cVector, input);
            while (i < distances.size() && distances.get(i) <= distance) {
                i++;
            }
            distances.add(i, distance);
            vectors.add(i, cVector);
        }

        return new ArrayList<>(vectors.subList(0, Math.min(k, vectors.size())));
    }

    /**
     * Calculates the distance between two {@link CharacteristicVector} objects
     * based on
     * the specified distance metric.
     *
     * @param v1 the first vector.
     * @param v2 the second vector.
     * @return the calculated distance between v1 and v2.
     * @throws MathUtilsException if an error occurs while calculating the distance.
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
     * Determines the most common label among a list of nearest neighbors.
     *
     * @param nearestNeighbors the list of nearest {@link CharacteristicVector}
     *                         neighbors.
     * @return the label with the highest count among the nearest neighbors.
     */
    private String majorityVote(ArrayList<CharacteristicVector> nearestNeighbors) {
        logger.debug("Performing majority vote among {} nearest neighbors", nearestNeighbors.size());
        HashMap<String, Integer> labelCount = new HashMap<>();

        // Count occurrences of each label in the k-nearest neighbors
        for (CharacteristicVector neighbor : nearestNeighbors) {
            String label = neighbor.getLabel();
            labelCount.put(label, labelCount.getOrDefault(label, 0) + 1);
        }

        logger.debug("Neighbor label counts: {}", labelCount);

        // Find the label with the maximum count
        String predictedLabel = null;
        int maxCount = 0;

        for (Map.Entry<String, Integer> entry : labelCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                predictedLabel = entry.getKey();
            }
        }

        logger.info("Selected label '{}' as majority vote with {} votes", predictedLabel, maxCount);

        return predictedLabel;
    }
}
