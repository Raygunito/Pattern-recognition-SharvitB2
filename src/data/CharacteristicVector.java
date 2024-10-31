package data;

/**
 * CharacteristicVector
 */
public class CharacteristicVector {
    double[] vector;
    String label;
    String method;
    String sample;

    public CharacteristicVector(double[] array, String label, String method, String sample) {
        this.vector = array;
        this.label = label;
        this.method = method;
        this.sample = sample;
    }

    public int getVectorSize() {
        return vector.length;
    }

    public double[] getVector() {
        return vector;
    }

    public String getLabel() {
        return label;
    }

    public String getMethod() {
        return method;
    }

    public String getSample() {
        return sample;
    }
    
    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();
        out.append("Method : ").append(method == null ? "Unknown" : method)
                .append(", Label : ").append(label == null ? "Unknown" : label)
                .append(" (").append(EntityConstants.getEntityByLabelCode(label)).append(")")
                .append(", Sample : ").append(label == null ? "Unknown" : sample).append("\n");

        if (vector == null || vector.length == 0) {
            out.append("No vector data available");
        } else {
            for (double value : vector) {
                out.append(value).append(", ");
            }
            out.setLength(out.length() - 2);
        }
        out.append("\n");
        return out.toString();
    }
}