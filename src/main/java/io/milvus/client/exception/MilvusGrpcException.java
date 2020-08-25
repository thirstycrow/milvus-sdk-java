package io.milvus.client.exception;

import io.grpc.StatusRuntimeException;

public class MilvusGrpcException extends MilvusException {

  public MilvusGrpcException(String host, StatusRuntimeException cause) {
    super(host, false, null, cause);
  }

  @Override
  public StatusRuntimeException getCause() {
    return (StatusRuntimeException) super.getCause();
  }
}
