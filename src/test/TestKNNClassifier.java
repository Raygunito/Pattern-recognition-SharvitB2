package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.Before;
import org.junit.Test;

import data.CharacteristicVector;
import process.KNNClassifier;

public class TestKNNClassifier {
    private KNNClassifier knnEuclidean;
    private KNNClassifier knnManhattan;
    private KNNClassifier knnMinkowski;
    private ArrayList<CharacteristicVector> trainingData;

    @Before
    public void setUp() {
        knnEuclidean = new KNNClassifier(3, KNNClassifier.EUCLIDEAN);
        knnManhattan = new KNNClassifier(3, KNNClassifier.MANHATTAN);
        knnMinkowski = new KNNClassifier(3, KNNClassifier.MINKOWSKI, 3); // norm = 3 for Minkowski

        trainingData = new ArrayList<>(Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 1.0 }, "A", null, null),
                new CharacteristicVector(new double[] { 2.0, 2.0 }, "A", null, null),
                new CharacteristicVector(new double[] { 3.0, 3.0 }, "B", null, null),
                new CharacteristicVector(new double[] { 6.0, 6.0 }, "B", null, null)));

        knnEuclidean.train(trainingData);
        knnManhattan.train(trainingData);
        knnMinkowski.train(trainingData);
    }

    @Test
    public void testPredictEuclidean() {
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.5, 1.5 }, null, null, null);
        String prediction = knnEuclidean.predict(input);
        assertEquals("A", prediction);
    }

    @Test
    public void testPredictManhattan() {
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.5, 1.5 }, null, null, null);
        String prediction = knnManhattan.predict(input);
        assertEquals("A", prediction);
    }

    @Test
    public void testPredictMinkowski() {
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.5, 1.5 }, null, null, null);
        String prediction = knnMinkowski.predict(input);
        assertEquals("A", prediction);
    }

    @Test
    public void testPredictWithNoTrainingData() {
        KNNClassifier knnNoData = new KNNClassifier(3, KNNClassifier.EUCLIDEAN);
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.0, 1.0 }, null, null, null);
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            knnNoData.predict(input);
        });
        assertEquals("Training data not set. Call train() before predict().", exception.getMessage());
    }

    @Test
    public void testTieBreakingInMajorityVote() {
        ArrayList<CharacteristicVector> tiedData = new ArrayList<>(Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 1.0 }, "A", null, null),
                new CharacteristicVector(new double[] { 2.0, 2.0 }, "A", null, null),
                new CharacteristicVector(new double[] { 3.0, 3.0 }, "B", null, null),
                new CharacteristicVector(new double[] { 4.0, 4.0 }, "B", null, null)));

        knnEuclidean.train(tiedData);
        CharacteristicVector input = new CharacteristicVector(new double[] { 2.5, 2.5 }, null, null, null);
        String prediction = knnEuclidean.predict(input);

        boolean isTieResultValid = prediction.equals("A") || prediction.equals("B");
        assertTrue("Expected prediction to be 'A' or 'B', but got: " + prediction, isTieResultValid);
    }

    @Test
    public void testInvalidDistanceMetric() {
        KNNClassifier knnInvalidMetric = new KNNClassifier(3, "invalid_metric");
        knnInvalidMetric.train(trainingData);
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.5, 1.5 }, null, null, null);
        String result = knnInvalidMetric.predict(input);
        assertEquals("Invalid metric should return maximum double value.", "B", result);
    }

    @Test
    public void testSingleNeighborPrediction() {
        KNNClassifier knnSingleNeighbor = new KNNClassifier(1, KNNClassifier.EUCLIDEAN);
        knnSingleNeighbor.train(trainingData);
        CharacteristicVector input = new CharacteristicVector(new double[] { 2.5, 2.5 }, null, null, null);
        String prediction = knnSingleNeighbor.predict(input);
        assertEquals("B", prediction); // Nearest single neighbor is "B"
    }

    @Test
    public void testKGreaterThanTrainingSize() {
        KNNClassifier knnLargeK = new KNNClassifier(10, KNNClassifier.EUCLIDEAN);
        knnLargeK.train(trainingData); // Only 4 training samples available
        CharacteristicVector input = new CharacteristicVector(new double[] { 2.5, 2.5 }, null, null, null);
        String prediction = knnLargeK.predict(input);
        assertTrue("Expected prediction to handle large k gracefully",
                prediction.equals("A") || prediction.equals("B"));
    }
}
