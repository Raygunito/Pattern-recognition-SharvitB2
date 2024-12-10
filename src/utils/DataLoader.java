package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;

import data.CharacteristicVector;
import logger.LoggerUtil;

/**
 * Utility class for loading and extracting {@code CharacteristicVector} data
 * from files and directories.
 * Provides functionality to read numerical data from text files, interpret
 * metadata from filenames,
 * and organize the extracted data into CharacteristicVector objects.
 */
public class DataLoader {
    /**
     * Array of predefined method identifiers used to determine the method from
     * filenames.
     */
    private static final String[] METHOD = { "ART", "ZRK", "E34", "GFD", "YNG" };
    private static final Logger logger = LoggerUtil.getLogger(DataLoader.class, Level.INFO);

    /**
     * Extracts a CharacteristicVector from a single file by reading numerical
     * values
     * and parsing metadata from the filename.
     *
     * @param pathname the path of the file to extract data from.
     * @return a CharacteristicVector object containing the extracted data and
     *         metadata.
     */
    public static CharacteristicVector extractFromFile(String pathname) {
        List<Double> list = new ArrayList<Double>();
        BufferedReader buffer = null;
        logger.debug("Starting extraction from file: {}", pathname);

        try {
            buffer = new BufferedReader(new FileReader(pathname));
            String line;
            while ((line = buffer.readLine()) != null) {
                try {
                    list.add(Double.parseDouble(line));
                } catch (Exception e) {
                    logger.warn("Couldn't convert '{}' to double in file: {}", line, pathname);
                }
            }
        } catch (IOException e) {
            logger.error("Error reading file {}: {}", pathname, e.getMessage());
        } finally {
            if (buffer != null) {
                try {
                    buffer.close();
                    logger.debug("File {} successfully closed.", pathname);
                } catch (IOException e) {
                    logger.error("Error closing file {}: {}", pathname, e.getMessage());
                }
            }
        }
        double[] arrayDoubles = new double[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arrayDoubles[i] = list.get(i);
        }
        String filename = Paths.get(pathname).getFileName().toString();
        String method = getMethodUsed(filename);
        String label = getLabelNumber(filename);
        String sample = getSampleNumber(filename);

        logger.debug("Extracted CharacteristicVector from file: {}, Method: {}, Label: {}, Sample: {}", pathname,
                method,
                label, sample);
        return new CharacteristicVector(arrayDoubles, label, method, sample);
    }

    /**
     * Extracts a list of CharacteristicVector objects from all valid files within a
     * folder.
     *
     * @param folderPath the path of the folder containing files to process.
     * @return an ArrayList of CharacteristicVector objects.
     */
    public static ArrayList<CharacteristicVector> extractFromFolder(String folderPath) {
        ArrayList<CharacteristicVector> vectors = new ArrayList<>();
        DirectoryStream<Path> stream = null;
        logger.info("Starting extraction from folder: {}", folderPath);
        try {
            stream = Files.newDirectoryStream(Paths.get(folderPath));
            for (Path filePath : stream) {
                if (Files.isRegularFile(filePath)) {
                    logger.debug("Processing file: {}", filePath.toString());
                    CharacteristicVector vector = extractFromFile(filePath.toString());
                    vectors.add(vector);
                }
            }
            logger.info("Finished extraction from folder: {}. Total files processed: {}", folderPath, vectors.size());
        } catch (IOException e) {
            logger.error("Error processing folder {}: {}", folderPath, e.getMessage());
        } finally {
            if (stream != null) {
                try {
                    stream.close();
                    logger.debug("DirectoryStream for folder {} successfully closed.", folderPath);
                } catch (IOException e) {
                    logger.error("Error closing DirectoryStream for folder {}: {}", folderPath, e.getMessage());
                }
            }
        }

        return vectors;
    }

    private static String getMethodUsed(String filename) {
        for (String methodString : METHOD) {
            if (filename.toLowerCase().contains(methodString.toLowerCase())) {
                logger.debug("Method identified as '{}' for file: {}", methodString, filename);
                return methodString;
            }
        }
        logger.warn("Method not identified in filename: {}", filename);
        return "Unknown";
    }

    private static String getLabelNumber(String filename) {
        if (filename.startsWith("s")) {
            String label = filename.substring(1, 3);
            logger.debug("Label identified as '{}' for file: {}", label, filename);
            return label;
        }
        logger.warn("Label not identified in filename: {}", filename);
        return "Unknown";
    }

    private static String getSampleNumber(String filename) {
        if (filename.startsWith("s")) {
            String sample = filename.substring(5, 7);
            logger.debug("Sample number identified as '{}' for file: {}", sample, filename);
            return sample;
        }
        logger.warn("Sample number not identified in filename: {}", filename);
        return "Unknown";
    }
}
