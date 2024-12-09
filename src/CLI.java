import java.util.ArrayList;
import java.util.Set;

import data.CharacteristicVector;
import data.Classifier;
import data.ConfusionMatrix;
import data.EntityConstants;
import process.KMeansClassifier;
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
        boolean cut = (nbClass == 10 ? true : false);
        int nbFold = 10;
        String[] location = { ART_FILE_LOCATION, GFD_FILE_LOCATION, YNG_FILE_LOCATION, ZRK_FILE_LOCATION };
        for (String stringPath : location) {
            System.out.println("Doing KMeans to folder " + stringPath);
            System.out.println("Manhattan");
            doKMeans(stringPath, nbClass * nbEchantillon, Classifier.MANHATTAN, cut,
            nbFold);
            System.out.println("Euclidean");
            doKMeans(stringPath, nbClass * nbEchantillon, Classifier.EUCLIDEAN, cut,
            nbFold);

            System.out.println("Doing KNN to folder " + stringPath);
            System.out.println("Manhattan");
            doKNN(stringPath, nbClass * nbEchantillon, Classifier.MANHATTAN, cut);
            System.out.println("Euclidean");
            doKNN(stringPath, nbClass * nbEchantillon, Classifier.EUCLIDEAN, cut);
            System.out.println("Minkowski p=3");
            doKNN(stringPath, nbClass * nbEchantillon, Classifier.MINKOWSKI, cut);
            doPRCurveKNN(stringPath);
        }
    }

    public static void doKMeans(String folderLocation, int datasetSize, String distanceMetric, boolean b, int nbFold) {
        ArrayList<CharacteristicVector> dataset = new ArrayList<>(
                DataLoader.extractFromFolder(folderLocation).subList(0, datasetSize));
        dataset = MachineLearningUtils.normalizeCharacteristicVectors(dataset);
        ArrayList<ArrayList<CharacteristicVector>> folds = MachineLearningUtils.createKFolds(dataset, nbFold);

        ArrayList<Double> sseArray = new ArrayList<>();
        ArrayList<Double> silArray = new ArrayList<>();

        // number of cluster to find the best K
        for (int k = 2; k < 19; k++) {
            KMeansClassifier kMeansClassifier = new KMeansClassifier(k, distanceMetric);
            if (distanceMetric == Classifier.MINKOWSKI) {
                kMeansClassifier = new KMeansClassifier(k, distanceMetric, 3);
            }
            double currentBestSSE = Double.MAX_VALUE;
            double currentBestSilScore = Double.MIN_VALUE;
            for (int i = 0; i < folds.size(); i++) {

                ArrayList<CharacteristicVector> testSet = folds.get(i);
                ArrayList<CharacteristicVector> trainSet = new ArrayList<>();
                for (int j = 0; j < folds.size(); j++) {
                    if (j != i) {
                        trainSet.addAll(folds.get(j));
                    }
                }

                kMeansClassifier.train(trainSet);
                double foldSSE = kMeansClassifier.calculateSSE();
                double foldSilScore = kMeansClassifier.calculateSilhouetteScore();
                if (foldSSE <= currentBestSSE) {
                    currentBestSSE = foldSSE;
                }

                if (foldSilScore >= currentBestSilScore) {
                    currentBestSilScore = foldSilScore;
                }

                // for (CharacteristicVector characteristicVector : testSet) {
                // String res = kMeansClassifier.predict(characteristicVector);
                // // System.out.println(characteristicVector.getLabel() + " give : " + res);
                // }
            }
            sseArray.add(Double.valueOf(currentBestSSE));
            silArray.add(Double.valueOf(currentBestSilScore));
        }
        System.out.print("SSE= ");
        for (int i = 0; i < sseArray.size(); i++) {
            String tmp = String.format("%.4f", sseArray.get(i));
            tmp = tmp.replace(',', '.');
            System.out.print(tmp+", ");
        }
        System.out.print("\nSilhouette= ");
        for (int i = 0; i < silArray.size(); i++) {
            String tmp = String.format("%.4f", silArray.get(i));
            tmp = tmp.replace(',', '.');
            System.out.print(tmp+", ");
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
        dataset = MachineLearningUtils.normalizeCharacteristicVectors(dataset);
        // Should we shuffle since we LOOCV inherit the fact that we check for each
        // point ?
        // Collections.shuffle(dataset);

        int bestK = 1;
        double bestAccuracy = Double.MIN_VALUE;
        ArrayList<ArrayList<Double>> prCurve = new ArrayList<>();
        prCurve.add(new ArrayList<Double>());
        prCurve.add(new ArrayList<Double>());
        for (int k = 1; k < 13; k++) {
            ConfusionMatrix testCfMatrix = new ConfusionMatrix(entites);
            KNNClassifier knn = new KNNClassifier(k, distanceMetric);
            if (distanceMetric == Classifier.MINKOWSKI) {
                knn = new KNNClassifier(k, distanceMetric, 3);
            }
            knn.train(dataset);
            MachineLearningUtils.performLOOCV(dataset, knn, testCfMatrix);
            double res = testCfMatrix.accuracy() * 100;
            // double res = MachineLearningUtils.performLOOCV(dataset, knn) * 100;
            if (res > bestAccuracy) {
                bestAccuracy = res;
                bestK = k;
            }
            System.out.println("k=" + k + ",F1-Score : " + String.format("%.2f", res) +
                    "%");
        }

        KNNClassifier classifier = new KNNClassifier(bestK, distanceMetric);
        if (distanceMetric == Classifier.MINKOWSKI) {
            classifier = new KNNClassifier(bestK, distanceMetric, 3);
        }
        System.out.println("BestK = " + bestK);
        classifier.train(dataset);
        MachineLearningUtils.performLOOCV(dataset, classifier, cfx);
        // the confusion matrix
        cfx.display();
        // Accuracy, Recall, Precision, F1 score
        cfx.showPerformance();
    }

    public static void doPRCurveKNN(String path) {
        ArrayList<CharacteristicVector> dataset = new ArrayList<>(
                DataLoader.extractFromFolder(path).subList(0, 120));
        ArrayList<ArrayList<CharacteristicVector>> fold = MachineLearningUtils.createKFolds(dataset, 12);
        ArrayList<CharacteristicVector> testFold = fold.get(0);
        ArrayList<CharacteristicVector> trainFold = new ArrayList<>() ;
        
        double[] precision = new double[12];
        double[] recall = new double[12];

        // exclude 1st fold (that we use as test fold)
        for (int i = 1; i < fold.size(); i++) {
            trainFold.addAll(fold.get(i));
        }

        KNNClassifier knn = new KNNClassifier(3, Classifier.EUCLIDEAN);
        knn.train(trainFold);

        for (CharacteristicVector cVector : testFold) {
            String trueLabel = cVector.getLabel();
            String res = knn.predict(cVector);
            ArrayList<CharacteristicVector> neigh = knn.getNeighbors(cVector, 12);

            int relevant = 0;
            int totalInClass = count(trainFold, trueLabel);
            
            for (int i = 0; i < neigh.size(); i++) {
                if (neigh.get(i).getLabel().equals(res)) {
                    relevant++;
                }

                precision[i] += (double)relevant/(double)(i+1);
                recall[i] += (double)relevant/totalInClass;
            }

        }

        for (int i = 0; i < precision.length; i++) {
            precision[i] /= testFold.size();
            recall[i] /= testFold.size();
        }

        String tmp;
        System.out.println("PR curve :");
        System.out.println("Recall : ");
        for (int i = 0; i < recall.length; i++) {
            tmp = String.valueOf(recall[i]).replace(',', '.');
            System.out.printf(tmp+",");
        }
        System.out.println();
        System.out.println("Precision : ");
        for (int i = 0; i < precision.length; i++) {
            tmp = String.valueOf(precision[i]).replace(',', '.');
            System.out.printf(tmp+",");
        }
    }

    private static int count(ArrayList<CharacteristicVector> dataset, String label) {
        int count = 0;
        for (CharacteristicVector vector : dataset) {
            if (vector.getLabel().equals(label)) {
                count++;
            }
        }
        return count;
    }
}
