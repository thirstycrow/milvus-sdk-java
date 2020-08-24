package io.milvus.client.exception;

public class MilvusClientException extends RuntimeException {
  @Override
  public synchronized Throwable fillInStackTrace() {
    return this;
  }
}
