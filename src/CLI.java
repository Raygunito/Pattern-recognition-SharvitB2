import java.util.ArrayList;
import java.util.Set;

import data.CharacteristicVector;
import data.Classifier;
import data.ConfusionMatrix;
import data.EntityConstants;
import process.KNNClassifier;
import utils.DataLoader;
import utils.MachineLearningUtils;

public class CLI {
    private static final String ART_FILE_LOCATION = "res\\Signatures\\ART";
    @SuppressWarnings("unused")
    private static final String E34_FILE_LOCATION = "res\\Signatures\\E34";
    private static final String GFD_FILE_LOCATION = "res\\Signatures\\GFD";
    private static final String YNG_FILE_LOCATION = "res\\Signatures\\Yang";
    private static final String ZRK_FILE_LOCATION = "res\\Signatures\\Zernike7";

    public static void main(String[] args) {
        int nbClass = 10;
        int nbEchantillon = 12;
        String[] location = { ART_FILE_LOCATION, GFD_FILE_LOCATION, YNG_FILE_LOCATION, ZRK_FILE_LOCATION };
        for (String stringPath : location) {
            System.out.println("Doing KNN to folder " + stringPath);
            doKNN(stringPath, nbClass * nbEchantillon, Classifier.EUCLIDEAN, true);
        }

    }

    public static void doKNN(String folderLocation, int datasetSize, String distanceMetric,
            boolean cutDatasetTo10Classes) {
        Set<String> entites = EntityConstants.entities.keySet();
        if (cutDatasetTo10Classes) {
            // Data set containt 18 classes, 12 sample each so we work on only the first 10
            // classes for the project
            entites.remove("11");
            entites.remove("12");
            entites.remove("13");
            entites.remove("14");
            entites.remove("15");
            entites.remove("16");
            entites.remove("17");
            entites.remove("18");
        }
        ConfusionMatrix cfx = new ConfusionMatrix(entites);
        ArrayList<CharacteristicVector> dataset = new ArrayList<>(
                DataLoader.extractFromFolder(folderLocation).subList(0, datasetSize));

        // Should we shuffle since we LOOCV inherit the fact that we check for each point ?
        // Collections.shuffle(dataset);

        int bestK = 1;
        double bestF1Score = Double.MIN_VALUE;
        for (int k = 1; k < 12; k++) {
            ConfusionMatrix testCfMatrix = new ConfusionMatrix(entites);
            KNNClassifier knn = new KNNClassifier(k, distanceMetric);
            knn.train(dataset);
            MachineLearningUtils.performLOOCV(dataset, knn, testCfMatrix);
            double res = testCfMatrix.globalF1Score() * 100;
            // double res = MachineLearningUtils.performLOOCV(dataset, knn) * 100;
            if (res > bestF1Score) {
                bestF1Score = res;
                bestK = k;
            }
            System.out.println("k=" + k + ",F1-Score : " + String.format("%.2f", res) + "%");
        }

        KNNClassifier classifier = new KNNClassifier(bestK, distanceMetric);
        classifier.train(dataset);
        MachineLearningUtils.performLOOCV(dataset, classifier, cfx);

        cfx.display();
        cfx.showPerformance();
    }
}
