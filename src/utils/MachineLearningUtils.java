package utils;

import java.util.ArrayList;
import java.util.Collections;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;

import data.CharacteristicVector;
import data.Classifier;
import data.ConfusionMatrix;
import data.EntityConstants;
import logger.LoggerUtil;

public class MachineLearningUtils {
    private static final Logger logger = LoggerUtil.getLogger(MachineLearningUtils.class, Level.INFO);

    /**
     * Performs k-fold cross-validation on the given dataset.
     *
     * @param dataset The dataset to be split into k folds.
     * @param k       The number of folds.
     * @return A list of k folds, each containing a list of CharacteristicVectors.
     */
    public static ArrayList<ArrayList<CharacteristicVector>> createKFolds(
            ArrayList<CharacteristicVector> dataset, int k) {
        ArrayList<ArrayList<CharacteristicVector>> folds = new ArrayList<>();
        ArrayList<CharacteristicVector> shuffled = new ArrayList<>(dataset);
        Collections.shuffle(shuffled);

        for (int i = 0; i < k; i++) {
            folds.add(new ArrayList<>());
        }

        for (int i = 0; i < shuffled.size(); i++) {
            folds.get(i % k).add(shuffled.get(i));
        }

        return folds;
    }

    /**
     * Performs Leave-One-Out Cross-Validation (LOOCV) on the given dataset with a
     * specified classifier.
     * Do not split your dataset (i.e. no 80/10/10 or anything else)
     * 
     * @param dataset    The dataset to be split into LOOCV folds.
     * @param classifier The classifier to be used for training and validation.
     * @return The accuracy of the classifier over all folds.
     */
    public static double performLOOCV(ArrayList<CharacteristicVector> dataset, Classifier classifier) {
        int correctPredictions = 0;
        for (int i = 0; i < dataset.size(); i++) {
            // Split dataset into training and validation set
            ArrayList<CharacteristicVector> trainingSet = new ArrayList<>(dataset);
            CharacteristicVector validationSet = trainingSet.remove(i);

            classifier.train(trainingSet);

            String predictedLabel = classifier.predict(validationSet);
            // Compare predicted label with actual label
            if (predictedLabel.equals(validationSet.getLabel())) {
                correctPredictions++;
            }
        }

        return (double) correctPredictions / dataset.size();
    }

    public static void performLOOCV(ArrayList<CharacteristicVector> dataset, Classifier classifier,
            ConfusionMatrix cfx) {
        if (dataset == null || dataset.isEmpty()) {
            logger.error("Dataset is null or empty. LOO-CV cannot proceed.");
            return;
        }
        logger.debug("Starting Leave-One-Out Cross-Validation (LOO-CV) with classifier: {}",
                classifier.getClass().getSimpleName());
        logger.debug("Dataset size: {}", dataset.size());
        for (int i = 0; i < dataset.size(); i++) {
            // Split dataset into training and validation set
            ArrayList<CharacteristicVector> trainingSet = new ArrayList<>(dataset);
            CharacteristicVector validationSet = trainingSet.remove(i);
            String validationLabel = validationSet.getLabel();
            classifier.train(trainingSet);

            String predictedLabel = classifier.predict(validationSet);
            logger.debug("Processed {}th point. Actual: {} ({}) | Predicted: {} ({})",
                    i + 1,
                    validationLabel,
                    EntityConstants.getEntityByLabelCode(validationLabel),
                    predictedLabel,
                    EntityConstants.getEntityByLabelCode(predictedLabel));
            // Compare predicted label with actual label
            cfx.increment(validationLabel, predictedLabel);
        }
    }

    public static ArrayList<CharacteristicVector> normalizeCharacteristicVectors(ArrayList<CharacteristicVector> vectorArray){
        return normalizeCharacteristicVectors(vectorArray,0,1);
    }
    public static ArrayList<CharacteristicVector> normalizeCharacteristicVectors(ArrayList<CharacteristicVector> vectorArray, int min, int max){
        if (vectorArray == null || vectorArray.isEmpty()) {
            return new ArrayList<>();
        }

        double globalMin = Double.MAX_VALUE;
        double globalMax = Double.MIN_VALUE;

        for (CharacteristicVector cv : vectorArray) {
            for (double value : cv.getVector()) {
                if (value < globalMin) globalMin = value;
                if (value > globalMax) globalMax = value;
            }
        }

        // Step 2: Normalize all vectors using the global min and max
        ArrayList<CharacteristicVector> normalizedVectors = new ArrayList<>();

        for (CharacteristicVector cv : vectorArray) {
            double[] originalVector = cv.getVector();
            double[] normalizedVector = new double[originalVector.length];

            for (int i = 0; i < originalVector.length; i++) {
                if (globalMax - globalMin == 0) {
                    // division by zero
                    normalizedVector[i] = (min + max) / 2.0;
                } else {
                    normalizedVector[i] = min + (originalVector[i] - globalMin) * (max - min) / (globalMax - globalMin);
                }
            }
            normalizedVectors.add(new CharacteristicVector(normalizedVector, cv.getLabel(), cv.getMethod(), cv.getSample()));
        }

        return normalizedVectors;
    }

}
