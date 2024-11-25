package test;

import data.ConfusionMatrix;
import org.junit.Before;
import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.assertEquals;

public class TestConfusionMatrix {

    private ConfusionMatrix confusionMatrix;

    @Before
    public void setUp() {
        Set<String> labels = Set.of("Class01", "Class02", "Class03");
        confusionMatrix = new ConfusionMatrix(labels);
    }

    @Test
    public void testIncrementAndGet() {
        confusionMatrix.increment("Class01", "Class01");
        confusionMatrix.increment("Class01", "Class03");
        confusionMatrix.increment("Class02", "Class01");

        assertEquals(1, confusionMatrix.get("Class01", "Class01"));
        assertEquals(1, confusionMatrix.get("Class01", "Class03"));
        assertEquals(1, confusionMatrix.get("Class02", "Class01"));
        assertEquals(0, confusionMatrix.get("Class03", "Class03"));

        // Verify invalid keys
        assertEquals(-1, confusionMatrix.get("Invalid", "Class01"));
        assertEquals(-1, confusionMatrix.get("Class01", "Invalid"));
    }

    @Test
    public void testAccuracy() {
        // Set up a scenario with some predictions
        confusionMatrix.increment("Class01", "Class01");
        confusionMatrix.increment("Class01", "Class03");
        confusionMatrix.increment("Class02", "Class01");
        confusionMatrix.increment("Class03", "Class03");

        // Total predictions = 4, correct predictions = 2
        assertEquals(0.5, confusionMatrix.accuracy(), 0.0001);
    }

    @Test
    public void testGlobalPrecision() {
        // Increment counts
        confusionMatrix.increment("Class01", "Class01"); // TP for Class01
        confusionMatrix.increment("Class02", "Class01"); // FP for Class01
        confusionMatrix.increment("Class03", "Class03"); // TP for Class03

        // Expected precision:
        // Class01: TP / (TP + FP) = 1 / (1 + 1) = 0.5
        // Class03: TP / (TP + FP) = 1 / (1 + 0) = 1.0
        // Class02: TP / (TP + FP) = 0 / (0 + 0) = 0 (if no FP, result 0 for label)

        // Average precision = (0.5 + 0 + 1) / 3 = 0.5
        assertEquals(0.5, confusionMatrix.globalPrecision(), 0.0001);
    }

    @Test
    public void testGlobalRecall() {
        // Increment counts
        confusionMatrix.increment("Class01", "Class01"); // TP for Class01
        confusionMatrix.increment("Class01", "Class03"); // FN for Class01
        confusionMatrix.increment("Class03", "Class03"); // TP for Class03

        // Expected recall:
        // Class01: TP / (TP + FN) = 1 / (1 + 1) = 0.5
        // Class03: TP / (TP + FN) = 1 / (1 + 0) = 1.0
        // Class02: TP / (TP + FN) = 0 / (0 + 0) = 0

        // Average recall = (0.5 + 0 + 1) / 3 = 0.5
        assertEquals(0.5, confusionMatrix.globalRecall(), 0.0001);
    }

    @Test
    public void testGlobalF1Score() {
        confusionMatrix.increment("Class01", "Class01"); // TP for Class01
        confusionMatrix.increment("Class01", "Class03"); // FN for Class01
        confusionMatrix.increment("Class02", "Class01"); // FP for Class01
        confusionMatrix.increment("Class03", "Class03"); // TP for Class03

        /*
         * P = colonne
         * R = ligne
         * A\P  01  02  03
         * 01   1   0   1
         * 02   1   0   0
         * 03   0   0   1
         * P(Class01) = 1/2  R(Class01) = 1/2   F1(01) = 2*(0.5 * 0.5) / (0.5+0.5) = 0.5 
         * P(Class02) = 0   R(Class02) = 0      F1(02) = 0
         * P(Class03) = 1/2   R(Class03) = 1      F1(03) = 2*(0.5 * 1) / (0.5 + 1) = 1/1.5 = 2/3
         * F1moy = 0.3333 / 0.8333 = 0.39997
         */
        double expectedF1 = 0.39997;
        assertEquals(expectedF1, confusionMatrix.globalF1Score(), 0.0001);
    }
}
