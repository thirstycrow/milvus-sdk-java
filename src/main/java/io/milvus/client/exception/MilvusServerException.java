package io.milvus.client.exception;

import io.milvus.grpc.ErrorCode;
import io.milvus.grpc.Status;

public class MilvusServerException extends RuntimeException {
  private ErrorCode errorCode;
  private String reason;

  public MilvusServerException(Status status) {
    this.errorCode = status.getErrorCode();
    this.reason = status.getReason();
  }

  public ErrorCode getErrorCode() {
    return errorCode;
  }

  public String getReason() {
    return reason;
  }

  @Override
  public synchronized Throwable fillInStackTrace() {
    return this;
  }

  @Override
  public String toString() {
    return String.format("MilvusServerException{errorCode=%s, reason=%s}", errorCode, reason);
  }
}
