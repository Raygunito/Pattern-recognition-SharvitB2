package data;

public class MathUtilsException extends Exception {
    public MathUtilsException() {
        super();
    }

    public MathUtilsException(String msg) {
        super(msg);
    }

    public MathUtilsException(String msg, Throwable err) {
        super(msg,err);
    }

}
