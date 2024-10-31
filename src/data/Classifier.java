package data;

import java.util.List;

public interface Classifier {
    public static final String EUCLIDEAN = "euclidean";
    public static final String MANHATTAN = "manhattan";
    public static final String MINKOWSKI = "minkowski";
    String predict(CharacteristicVector vector);
    
    default void train(List<CharacteristicVector> trainingData) {}
    
}
