package data;

import java.util.HashMap;
import java.util.Set;

/**
 * This class represents a Confusion Matrix for classification tasks.
 * The matrix stores counts of actual vs predicted classifications
 * and provides methods to calculate performance metrics such as accuracy,
 * precision, recall, and F1-score.
 */
public class ConfusionMatrix {
    /**
     * The outer map represents the actual labels (rows),
     * and the inner map represents the predicted labels (columns).
     */
    private HashMap<String, HashMap<String, Integer>> matrix;

    /**
     * Constructor to initialize the confusion matrix with given labels.
     * Initializes all counts to 0.
     *
     * @param labels The set of possible labels (classes) for the matrix.
     */
    public ConfusionMatrix(Set<String> labels) {
        matrix = new HashMap<>();
        for (String row : labels) {
            matrix.put(row, new HashMap<>());
            for (String column : labels) {
                matrix.get(row).put(column, 0);
            }
        }
    }

    /**
     * Increments the count of the corresponding actual vs predicted classification
     * in the matrix.
     *
     * @param actual    The actual class label.
     * @param predicted The predicted class label.
     */
    public void increment(String actual, String predicted) {
        if (matrix.containsKey(actual) && matrix.get(actual).containsKey(predicted)) {
            int currentCount = matrix.get(actual).get(predicted);
            matrix.get(actual).put(predicted, currentCount + 1);
        }
    }

    /**
     * Retrieves the count of the specific pair of actual and predicted labels.
     *
     * @param actual    The actual class label.
     * @param predicted The predicted class label.
     * @return The count of occurrences,
     *         or -1 if the labels are not found.
     */
    public int get(String actual, String predicted) {
        if (matrix.containsKey(actual) && matrix.get(actual).containsKey(predicted)) {
            return matrix.get(actual).get(predicted);
        }
        return -1;
    }

    /**
     * Display the confusion matrix in a readable format
     */
    public void display() {
        System.out.print("A\\P\t");
        for (String column : matrix.keySet()) {
            System.out.print(column + "\t");
        }
        System.out.println();

        for (String row : matrix.keySet()) {
            System.out.print(row + "\t");
            for (String column : matrix.get(row).keySet()) {
                System.out.print(matrix.get(row).get(column) + "\t");
            }
            System.out.println();
        }
    }

    /**
     * Calculates the accuracy of the classification based on the confusion matrix.
     * Accuracy is the ratio of correct predictions to total predictions.
     * (FR : aussi appelé Taux de reconnaissance)
     * 
     * @return The accuracy value between 0.0 and 1.0.
     */
    public double accuracy() {
        int correctPredictions = 0;
        int totalPredictions = 0;
        // Explore each [row][column] of the matrix
        for (String actual : matrix.keySet()) {
            for (String predicted : matrix.get(actual).keySet()) {
                int count = matrix.get(actual).get(predicted);
                totalPredictions += count;
                if (actual.equals(predicted)) {
                    correctPredictions += count;
                }
            }
        }

        if (totalPredictions == 0)
            return 0.0;
        return (double) correctPredictions / totalPredictions;
    }

    /**
     * Calculates the global precision across all classes in the confusion matrix.
     * Precision is the ratio of true positives to the sum of true positives and
     * false positives.
     * (FR : Précision)
     * 
     * @return The global precision value.
     */
    public double globalPrecision() {
        double totalPrecision = 0;
        int totalClass = matrix.keySet().size();
        for (String label : matrix.keySet()) {
            totalPrecision += precision(label);
        }

        if (totalPrecision == 0) {
            return 0;
        }

        return (double) totalPrecision / (double) totalClass;
    }

    private double precision(String label) {
        int truePositive = this.get(label, label);
        int falsePositive = 0;
        for (String row : matrix.keySet()) {
            if (!row.equals(label)) {
                falsePositive += this.get(row, label);
            }
        }

        if (truePositive == 0) {
            return 0;
        }
        return (double) truePositive / (double) (falsePositive + truePositive);
    }

    /**
     * Calculates the global recall across all classes in the confusion matrix.
     * Recall is the ratio of true positives to the sum of true positives and false
     * negatives.
     * (FR : Rappel)
     * 
     * @return The global recall value.
     */
    public double globalRecall() {
        double totalRecall = 0;
        int totalClass = matrix.keySet().size();
        for (String label : matrix.keySet()) {
            totalRecall += recall(label);
        }

        if (totalRecall == 0) {
            return 0;
        }

        return (double) totalRecall / (double) totalClass;
    }

    private double recall(String label) {
        int truePositive = this.get(label, label);
        // Inside all the false negative there is also the true positive inside that why
        // we substract it
        int falseNegative = matrix.get(label).values().stream().mapToInt(Integer::intValue).sum() - truePositive;
        return (double) truePositive / (double) (falseNegative + truePositive);
    }

    /**
     * Calculates the global F1-Score across all classes in the confusion matrix.
     * The F1-Score is the harmonic mean of precision and recall.
     *
     * @return The global F1-Score value.
     */
    public double globalF1Score() {
        double precision = globalPrecision();
        double recall = globalRecall();

        if (precision + recall == 0)
            return 0.0;
        return (double) (2 * (precision * recall)) / (double) (precision + recall);
    }
}
