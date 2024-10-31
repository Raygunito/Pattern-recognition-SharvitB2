package process;

import java.util.ArrayList;
import java.util.List;

import data.CharacteristicVector;
import data.Classifier;
import data.MathUtilsException;
import utils.MathUtils;

public class KMeansClassifier implements Classifier {

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

    public KMeansClassifier(int kCluster, int norm) {
        this.k = kCluster;
        this.distanceMetric = Classifier.MINKOWSKI;
        this.norm = norm;
    }

    @Override
    public void train(List<CharacteristicVector> trainingData) {
        this.trainData = new ArrayList<>(trainingData);
        // TODO : Choose centroid method
        arrayCentroid = new ArrayList<>();

        cluster = resetArrayCluster();

        boolean same = false;

        while (!same) {
            cluster = resetArrayCluster();
            
            for (CharacteristicVector cVector : trainingData) {
                double[] distanceFromCentroid = arrayDistanceFromCentroid(cVector, arrayCentroid);
                int indexToAdd = minIndex(distanceFromCentroid);
                cluster.get(indexToAdd).add(cVector);
            }
            ArrayList<CharacteristicVector> newCentroid = null;

            if (newCentroid == arrayCentroid) {
                same = true;
            }
            arrayCentroid = newCentroid;
        }
    }
    // TODO
    @Override
    public String predict(CharacteristicVector vector) throws IllegalStateException {
        if (trainData == null) {
            throw new IllegalStateException("Training data not set. Call train() before predict().");
        }
        return "";
    }

    private ArrayList<ArrayList<CharacteristicVector>> resetArrayCluster() {
        ArrayList<ArrayList<CharacteristicVector>> clust = new ArrayList<ArrayList<CharacteristicVector>>();
        for (int i = 0; i < k; i++) {
            clust.add(new ArrayList<CharacteristicVector>());
        }
        return clust;
    }

    private double[] arrayDistanceFromCentroid(CharacteristicVector cVector,
            ArrayList<CharacteristicVector> arrayCentroid) {
        double[] distanceCentroid = new double[arrayCentroid.size()];
        for (int i = 0; i < arrayCentroid.size(); i++) {
            distanceCentroid[i] = calculateDistance(cVector, arrayCentroid.get(i));
        }
        return distanceCentroid;
    }

    private double calculateDistance(CharacteristicVector v1, CharacteristicVector v2) {
        switch (distanceMetric) {
            case Classifier.EUCLIDEAN:
                try {
                    return MathUtils.distEuclidean(v1, v2);
                } catch (MathUtilsException e) {
                    e.printStackTrace();
                }
            case Classifier.MANHATTAN:
                try {
                    return MathUtils.distManhattan(v1, v2);
                } catch (MathUtilsException e) {
                    e.printStackTrace();
                }
                break;
            case Classifier.MINKOWSKI:
                try {
                    return MathUtils.distMinkowski(v1, v2, norm);
                } catch (MathUtilsException e) {
                    e.printStackTrace();
                }
                break;
            default:
                break;
        }
        return Double.MAX_VALUE;
    }

    private int minIndex(double[] array) {
        int i = 0;
        double min = Double.MAX_VALUE;
        for (int j = 0; j < array.length; j++) {
            if (array[j] < min) {
                i = j;
                min = array[j];
            }
        }
        return i;
    }
}
