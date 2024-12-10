package data;

/**
 * Custom exception class for mathematical utility errors.
 * Extends {@code Exception} to provide detailed error information.
 */
public class MathUtilsException extends Exception {
    public MathUtilsException() {
        super();
    }

    public MathUtilsException(String msg) {
        super(msg);
    }

    public MathUtilsException(String msg, Throwable err) {
        super(msg, err);
    }

}
