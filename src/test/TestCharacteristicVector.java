package test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import data.CharacteristicVector;

public class TestCharacteristicVector {

    private CharacteristicVector vector;

    @Before
    public void setUp() {
        double[] array = { 1.0, 2.0, 3.0 };
        vector = new CharacteristicVector(array, "Label1", "Method1", "Sample1");
    }

    @Test
    public void testGetVectorSize() {
        assertEquals(3, vector.getVectorSize());
    }

    @Test
    public void testGetVector() {
        double[] result = vector.getVector();
        assertArrayEquals(new double[] { 1.0, 2.0, 3.0 }, result, 0.0);
    }

    @Test
    public void testToString() {
        String expectedOutput = "Method : Method1, Label : Label1 (Unknown code), Sample : Sample1\n" +
                "1.0, 2.0, 3.0\n";
        assertEquals(expectedOutput, vector.toString());
    }

    @Test
    public void testToStringEmptyVector() {
        CharacteristicVector emptyVector = new CharacteristicVector(new double[] {}, "Label2", "Method2", "Sample2");
        String expectedOutput = "Method : Method2, Label : Label2 (Unknown code), Sample : Sample2\nNo vector data available\n";
        assertEquals(expectedOutput, emptyVector.toString());
    }

    @Test
    public void testToStringWithNullValues() {
        CharacteristicVector nullVector = new CharacteristicVector(new double[] { 1.0, 2.0 }, null, null, null);
        String expectedOutput = "Method : Unknown, Label : Unknown (Unknown code), Sample : Unknown\n" +
                "1.0, 2.0\n";
        assertEquals(expectedOutput, nullVector.toString());
    }
}
