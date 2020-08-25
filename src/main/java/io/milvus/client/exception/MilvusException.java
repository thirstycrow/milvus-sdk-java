package io.milvus.client.exception;

public class MilvusException extends RuntimeException {
  private String host;
  private boolean fillInStackTrace;

  MilvusException(String host, boolean fillInStackTrace) {
    this(host, fillInStackTrace, null, null);
  }

  public MilvusException(String host, Throwable cause) {
    this(host, false, null, cause);
  }

  MilvusException(String host, boolean fillInStackTrace, String message, Throwable cause) {
    super(message, cause);
    this.host = host;
    this.fillInStackTrace = fillInStackTrace;
  }

  MilvusException(boolean fillInStackTrace, String message) {
    super(message);
    this.fillInStackTrace = fillInStackTrace;
  }

  @Override
  public final String getMessage() {
    return String.format("%s: %s", host, getErrorMessage());
  }

  protected String getErrorMessage() {
    return super.getMessage();
  }

  @Override
  public synchronized Throwable fillInStackTrace() {
    return fillInStackTrace ? super.fillInStackTrace() : this;
  }
}
