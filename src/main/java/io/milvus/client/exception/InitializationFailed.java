package io.milvus.client.exception;

public class InitializationFailed extends MilvusClientException {
  private String host;
  private String reason;

  public InitializationFailed(String host, String reason) {
    this.host = host;
    this.reason = reason;
  }

  @Override
  public String getMessage() {
    return String.format("%s: Initialization failed: %s", host, reason);
  }
}
