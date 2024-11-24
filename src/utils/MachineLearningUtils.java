package utils;

import java.util.ArrayList;
import java.util.Collections;
import data.CharacteristicVector;
import data.Classifier;
import data.ConfusionMatrix;

public class MachineLearningUtils {

    /**
     * Performs k-fold cross-validation on the given dataset.
     *
     * @param dataset   The dataset to be split into k folds.
     * @param k         The number of folds.
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
     * Performs Leave-One-Out Cross-Validation (LOOCV) on the given dataset with a specified classifier.
     * Do not split your dataset (i.e. no 80/10/10 or anything else)
     * @param dataset   The dataset to be split into LOOCV folds.
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

    public static void performLOOCV(ArrayList<CharacteristicVector> dataset, Classifier classifier,ConfusionMatrix cfx) {
        for (int i = 0; i < dataset.size(); i++) {
            // Split dataset into training and validation set
            ArrayList<CharacteristicVector> trainingSet = new ArrayList<>(dataset);
            CharacteristicVector validationSet = trainingSet.remove(i);
            
            classifier.train(trainingSet);

            String predictedLabel = classifier.predict(validationSet);
            // Compare predicted label with actual label
            cfx.increment(validationSet.getLabel(), predictedLabel);
        }        
    }
}
