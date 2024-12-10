package data;

import java.util.List;

/**
 * An interface for implementing different classification algorithms.
 * Provides methods for training and predicting classes based on characteristic
 * vectors.
 */
public interface Classifier {

    /** Constant for the Euclidean distance metric. */
    public static final String EUCLIDEAN = "euclidean";

    /** Constant for the Manhattan distance metric. */
    public static final String MANHATTAN = "manhattan";

    /** Constant for the Minkowski distance metric. */
    public static final String MINKOWSKI = "minkowski";

    /**
     * Predicts the class label for a given characteristic vector.
     *
     * @param vector the {@code CharacteristicVector} to classify.
     * @return the predicted class label.
     */
    String predict(CharacteristicVector vector);

    /**
     * Trains the classifier with a given set of training data.
     * This default implementation does nothing and can be overridden by
     * implementations.
     *
     * @param trainingData the list of {@code CharacteristicVector} objects to train
     *                     the classifier.
     */
    default void train(List<CharacteristicVector> trainingData) {
    }
}
