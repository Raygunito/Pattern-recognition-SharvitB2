import java.util.ArrayList;

import data.CharacteristicVector;
import data.Classifier;
import data.EntityConstants;
import process.KNNClassifier;
import utils.DataLoader;

public class CLI {
    private static final String ART_FILE_LOCATION = "SharvitB2\\Signatures\\ART";
    private static final String E34_FILE_LOCATION = "SharvitB2\\Signatures\\E34";

    public static void main(String[] args) {
        KNNClassifier knnClassifier = new KNNClassifier(5, Classifier.EUCLIDEAN);
        ArrayList<CharacteristicVector> array = DataLoader.extractFromFolder(E34_FILE_LOCATION);
        CharacteristicVector cVector2 = array.remove(21);
        knnClassifier.train(array);
        String prediction = knnClassifier.predict(cVector2);
        System.out.println("Found : " + EntityConstants.getEntityByLabelCode(prediction));
    }
}
