package data;

import java.util.HashMap;
import java.util.Map;

/**
 * EntityConstants
 */
public class EntityConstants {
    public static final Map<String, String> entities = new HashMap<>();
    static {
        entities.put("01", "Pigeon");
        entities.put("02", "Os");
        entities.put("03", "Tapis");
        entities.put("04", "Chameau");
        entities.put("05", "Voiture simple");
        entities.put("06", "Humain");
        entities.put("07", "Voiture ancienne");
        entities.put("08", "Elephant");
        entities.put("09", "Visage");
        entities.put("10", "Fourche");
        entities.put("11", "Tombe funeraire");
        entities.put("12", "Verre a pied");
        entities.put("13", "Marteau");
        entities.put("14", "Coeur");
        entities.put("15", "Cle de voiture");
        entities.put("16", "Monstre");
        entities.put("17", "Raie");
        entities.put("18", "Tortue");
    }
    public static String getEntityByLabelCode(String code) {
        return entities.getOrDefault(code, "Unknown code");
    }
}