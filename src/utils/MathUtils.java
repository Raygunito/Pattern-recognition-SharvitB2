package utils;

import data.CharacteristicVector;
import data.MathUtilsException;

/**
 * The MathUtils class provides static methods to perform various mathematical
 * calculations
 * that are used throughout the project. These calculations include distance
 * metrics such as
 * Euclidean, Manhattan, and Minkowski distances between characteristic vectors.
 */
public class MathUtils {

    /**
     * Calculates the Euclidean distance between two characteristic vectors.
     * The Euclidean distance is computed using the formula:
     * 
     * <pre>
     * distance = sqrt(sum((vect1[i] - vect2[i]) ^ 2))
     * </pre>
     * 
     * @param vect1 the first characteristic vector
     * @param vect2 the second characteristic vector
     * @return the Euclidean distance between the two vectors
     * @throws MathUtilsException if the sizes of the two vectors are not the same
     */
    public static double distEuclidean(CharacteristicVector vect1, CharacteristicVector vect2)
            throws MathUtilsException {
        if (vect1.getVectorSize() != vect2.getVectorSize()) {
            throw new MathUtilsException("Vectors are not the same size !");
        }
        double sum = 0;
        int length = vect1.getVectorSize();
        for (int i = 0; i < length; i++) {
            sum += Math.pow(Math.abs(vect1.getVector()[i] - vect2.getVector()[i]), 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * Calculates the Manhattan distance between two characteristic vectors.
     * The Manhattan distance is computed using the formula:
     * 
     * <pre>
     *   distance = sum(|vect1[i] - vect2[i]|)
     * </pre>
     *
     * @param vect1 the first characteristic vector
     * @param vect2 the second characteristic vector
     * @return the Manhattan distance between the two vectors
     * @throws MathUtilsException if the sizes of the two vectors are not the same
     */
    public static double distManhattan(CharacteristicVector vect1, CharacteristicVector vect2)
            throws MathUtilsException {
        if (vect1.getVectorSize() != vect2.getVectorSize()) {
            throw new MathUtilsException("Vectors are not the same size !");
        }
        double sum = 0;
        int length = vect1.getVectorSize();
        for (int i = 0; i < length; i++) {
            sum += Math.abs(vect1.getVector()[i] - vect2.getVector()[i]);
        }
        return sum;
    }

    /**
     * Calculates the Minkowski distance between two characteristic vectors.
     * The Minkowski distance is defined as:
     * 
     * <pre>
     *   distance = (sum(|vect1[i] - vect2[i]|^p))^(1/p)
     * </pre>
     * 
     * @param vect1 the first characteristic vector
     * @param vect2 the second characteristic vector
     * @param p     the order of the norm (should be a positive integer)
     * @return the Minkowski distance between the two vectors
     * @throws MathUtilsException if the sizes of the two vectors are not the same
     */
    public static double distMinkowski(CharacteristicVector vect1, CharacteristicVector vect2, int p)
            throws MathUtilsException {
        if (p <= 0) {
            throw new MathUtilsException("Order of the norm should be a positive integer p=" + p);
        }
        if (vect1.getVectorSize() != vect2.getVectorSize()) {
            throw new MathUtilsException("Vectors are not the same size !");
        }
        double sum = 0;
        int length = vect1.getVectorSize();
        for (int i = 0; i < length; i++) {
            sum += Math.pow(Math.abs(vect1.getVector()[i] - vect2.getVector()[i]), p);
        }
        return Math.pow(sum, 1.0 / (double) p);
    }
}
