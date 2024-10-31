package logger;

import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.logging.log4j.core.config.builder.api.AppenderComponentBuilder;
import org.apache.logging.log4j.core.config.builder.api.ConfigurationBuilder;
import org.apache.logging.log4j.core.config.builder.api.ConfigurationBuilderFactory;
import org.apache.logging.log4j.core.config.builder.api.RootLoggerComponentBuilder;

/**
 * Utility class for configuring and retrieving Log4j2 loggers.
 * <p>
 * This class supports logging both in an IDE environment and in an exported
 * JAR, with customizable logging levels
 * and configuration paths. By default, it looks for {@code log4j2.properties}
 * for configuration, and if not found,
 * applies a built-in configuration that outputs to both console and a log file.
 * </p>
 * <p>
 * <strong>Note:</strong> When used in the exported JAR, logs will be generated
 * in a file named
 * {@value LoggerUtil#LOG_NAME} in the command line’s working directory.
 * </p>
 */
public class LoggerUtil {
    private static final String DEFAULT_CONFIG_NAME = "log4j2.properties";
    private static final String DEFAULT_CONFIG_PATH = "/logger/";
    private static final String LOG_NAME = "log.txt";

    private LoggerUtil() {
    }

    /**
     * Returns a default logger for the specified class,
     * using the default configuration path if available.
     * In case nothing is found we use the built-in configuration which logs to
     * console and file at Trace Level.
     * 
     * @param logClass the class for which the logger is created
     * @return the configured Logger instance
     */
    public static Logger getLogger(Class<?> logClass) {
        return getLogger(logClass,null);
    }

    /**
     * Retrieves a logger for the specified class and log level.
     *
     * @param logClass the class for which the logger is created
     * @param logLevel the logging level to set
     * @return the configured Logger instance
     */
    public static Logger getLogger(Class<?> logClass, Level logLevel) {
        return getLogger(logClass, logLevel, DEFAULT_CONFIG_PATH + DEFAULT_CONFIG_NAME);
    }

    /**
     * Retrieves a logger for the specified class, level, and configuration path.
     * The configuration file path should be relative to the class location if
     * within a JAR.
     *
     * @param logClass   the class for which the logger is created
     * @param logLevel   the logging level to set
     * @param configPath the relative path to the configuration file
     * @return the configured Logger instance
     */
    public static Logger getLogger(Class<?> logClass, Level logLevel, String configPath) {
        LoggerContext context = LoggerContext.getContext(false);
        Configuration cfg = context.getConfiguration();

        URL configFileURL = logClass.getResource(configPath);

        if (configFileURL != null) {
            applyConfigurationIfChanged(context, configFileURL, cfg);
        } else {
            System.out.println("Configuration file not found. Using default configuration.");
            setupDefaultConfiguration(context);
        }

        String className = logClass.getName();
        Logger logger = LogManager.getLogger(className);

        LoggerConfig lcfg = cfg.getLoggerConfig(className);
        // Example : Config log level : Trace, for MyClass.java i want Error log level
        // only
        if (logLevel != null && (logLevel.isMoreSpecificThan(lcfg.getLevel()) || logLevel.isLessSpecificThan(lcfg.getLevel()))) {
            Configurator.setLevel(className, logLevel);
        }

        return logger;
    }

    /**
     * Updates the logger configuration if the provided URI differs from the current
     * configuration.
     * 
     * @param context       the LoggerContext instance
     * @param configFileURL the URL of the new configuration file
     * @param config        the current Configuration instance
     */
    private static void applyConfigurationIfChanged(LoggerContext context, URL configFileURL,
            Configuration configuration) {
        try {
            URI currentConfigURI = configuration.getConfigurationSource().getURI();
            URI newConfigURI = configFileURL.toURI();

            // Set new config location if it's different from the current one (or not set
            // yet)
            if (currentConfigURI == null || !currentConfigURI.equals(newConfigURI)) {
                context.setConfigLocation(newConfigURI);
            }
        } catch (URISyntaxException e) {
            System.err.println("Error parsing configuration file URI: " + e.getMessage());
        }
    }

    /**
     * Sets up a default logging configuration with both console and file output.
     *
     * @param context     the LoggerContext instance
     * @param logFilePath the path to the log file
     */
    private static void setupDefaultConfiguration(LoggerContext context) {
        ConfigurationBuilder<?> builder = ConfigurationBuilderFactory.newConfigurationBuilder();

        // Console Appender Output to console
        AppenderComponentBuilder consoleAppender = builder.newAppender("ConsoleAppender", "CONSOLE")
                .addAttribute("target", org.apache.logging.log4j.core.appender.ConsoleAppender.Target.SYSTEM_OUT);
        consoleAppender.add(builder.newLayout("PatternLayout")
                .addAttribute("pattern", "%d{yyyy-MM-dd HH:mm:ss} [%t] %-5level %c{1} - %msg%n"));
        builder.add(consoleAppender);

        // File Appender Output to a file log.txt
        AppenderComponentBuilder fileAppender = builder.newAppender("FileAppender", "FILE")
                .addAttribute("fileName", LOG_NAME);
        fileAppender.add(builder.newLayout("PatternLayout")
                .addAttribute("pattern", "%d{yyyy-MM-dd HH:mm:ss} [%t] %-5level %c{1} - %msg%n"));
        builder.add(fileAppender);

        RootLoggerComponentBuilder rootLogger = builder.newRootLogger(Level.TRACE);
        rootLogger.add(builder.newAppenderRef("ConsoleAppender"));
        rootLogger.add(builder.newAppenderRef("FileAppender"));
        builder.add(rootLogger);

        context.start(builder.build());
    }
}
