package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import data.CharacteristicVector;
import data.Classifier;
import process.KMeansClassifier;

public class TestKMeansClassifier {
    private KMeansClassifier kMeansClassifier;

    @Before
    public void setUp() {
        int k = 3;
        kMeansClassifier = new KMeansClassifier(k, Classifier.EUCLIDEAN);
    }

    @Test
    public void testInitialization() {
        assertNotNull(kMeansClassifier);
    }

    @Test
    public void throwExceptionNoTrain() {
        KMeansClassifier noTrainKMeansClassifier = new KMeansClassifier(3, Classifier.EUCLIDEAN);
        CharacteristicVector input = new CharacteristicVector(new double[] { 1.0, 1.0 }, null, null, null);
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            noTrainKMeansClassifier.predict(input);
        });
        assertEquals("Training data not set. Call train() before predict().", exception.getMessage());
    }

    @Test
    public void testTraining() {
        List<CharacteristicVector> trainingData = Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label1", null, null),
                new CharacteristicVector(new double[] { 3.0, 4.0 }, "Label2", null, null),
                new CharacteristicVector(new double[] { 5.0, 6.0 }, "Label3", null, null),
                new CharacteristicVector(new double[] { 7.0, 8.0 }, "Label4", null, null));

        kMeansClassifier.train(trainingData);

        assertNotNull(kMeansClassifier.getCluster());
        assertEquals(3, kMeansClassifier.getCluster().size());
    }

    @Test
    public void testPredictAfterTraining() {
        List<CharacteristicVector> trainingData = Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label1", null, null),
                new CharacteristicVector(new double[] { 3.0, 4.0 }, "Label2", null, null),
                new CharacteristicVector(new double[] { 5.0, 6.0 }, "Label3", null, null),
                new CharacteristicVector(new double[] { 7.0, 8.0 }, "Label4", null, null));

        kMeansClassifier.train(trainingData);

        CharacteristicVector input = new CharacteristicVector(new double[] { 2.0, 3.0 }, null, null, null);
        String predictedCluster = kMeansClassifier.predict(input);

        assertNotNull(predictedCluster);
        assertTrue(predictedCluster.startsWith("Cluster "));
    }

    @Test
    public void testEmptyClusterHandling() {
        List<CharacteristicVector> trainingData = Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label1", null, null),
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label2", null, null));

        kMeansClassifier.train(trainingData);

        assertNotNull(kMeansClassifier.getCluster());
        assertTrue(kMeansClassifier.getCluster().stream().anyMatch(ArrayList::isEmpty));
    }

    @Test
    public void testSilhouetteScoreCalculation() {
        List<CharacteristicVector> trainingData = Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label1", null, null),
                new CharacteristicVector(new double[] { 3.0, 4.0 }, "Label2", null, null),
                new CharacteristicVector(new double[] { 5.0, 6.0 }, "Label3", null, null),
                new CharacteristicVector(new double[] { 7.0, 8.0 }, "Label4", null, null));

        kMeansClassifier.train(trainingData);
        double silhouetteScore = kMeansClassifier.calculateSilhouetteScore();

        assertTrue(silhouetteScore >= -1.0 && silhouetteScore <= 1.0);
    }

    @Test
    public void testSSECalculation() {
        List<CharacteristicVector> trainingData = Arrays.asList(
                new CharacteristicVector(new double[] { 1.0, 2.0 }, "Label1", null, null),
                new CharacteristicVector(new double[] { 3.0, 4.0 }, "Label2", null, null),
                new CharacteristicVector(new double[] { 5.0, 6.0 }, "Label3", null, null),
                new CharacteristicVector(new double[] { 7.0, 8.0 }, "Label4", null, null));

        kMeansClassifier.train(trainingData);
        double sse = kMeansClassifier.calculateSSE();

        assertTrue(sse >= 0);
    }
}
