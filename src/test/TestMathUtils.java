package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import data.CharacteristicVector;
import data.MathUtilsException;
import utils.MathUtils;

public class TestMathUtils {

    private CharacteristicVector vect1;
    private CharacteristicVector vect2;
    private CharacteristicVector vect3; // Different size vector to test exceptions

    @Before
    public void setUp() {
        double[] vector1 = { 1.0, 2.0, 3.0 };
        double[] vector2 = { 4.0, 5.0, 6.0 };
        double[] vector3 = { 7.0, 8.0 }; // Different size vector

        vect1 = new CharacteristicVector(vector1, "Label1", "Method1", "Sample1");
        vect2 = new CharacteristicVector(vector2, "Label2", "Method2", "Sample2");
        vect3 = new CharacteristicVector(vector3, "Label3", "Method3", "Sample3");
    }

    @Test
    public void testDistEuclidean() throws MathUtilsException {
        double result = MathUtils.distEuclidean(vect1, vect2);
        // Expected: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assertEquals(5.196, result, 0.001);
    }

    @Test
    public void testDistManhattan() throws MathUtilsException {
        double result = MathUtils.distManhattan(vect1, vect2);
        // Expected: |4-1| + |5-2| + |6-3| = 9
        assertEquals(9.0, result, 0.0);
    }

    @Test
    public void testDistMinkowski_p3() throws MathUtilsException {
        double result = MathUtils.distMinkowski(vect1, vect2, 3);
        // Expected: cube root of (|4-1|^3 + |5-2|^3 + |6-3|^3) = cube root of 81 =
        // 4.326
        assertEquals(4.326, result, 0.001);
    }

    @Test
    public void testDistMinkowski_p1() throws MathUtilsException {
        double result = MathUtils.distMinkowski(vect1, vect2, 1);
        // Minkowski with p=1 should behave like Manhattan distance
        assertEquals(9.0, result, 0.0);
    }

    @Test
    public void testDistMinkowski_pnegative3() throws MathUtilsException {
        MathUtilsException exception = assertThrows(MathUtilsException.class,
                () -> MathUtils.distMinkowski(vect1, vect2, -3));
        assertEquals("Order of the norm should be a positive integer p=-3", exception.getMessage());
    }

    @Test
    public void testVectorsNotSameSize() {
        try {
            MathUtils.distEuclidean(vect1, vect3);
            fail("Expected MathUtilsException not thrown");
        } catch (MathUtilsException e) {
            assertEquals("Vectors are not the same size !", e.getMessage());
        }
    }

    @Test
    public void testDistEuclideanSameVector() throws MathUtilsException {
        double result = MathUtils.distEuclidean(vect1, vect1);
        // The Euclidean distance between the same vector should be 0
        assertEquals(0.0, result, 0.0);
    }
}
