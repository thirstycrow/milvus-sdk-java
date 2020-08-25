package io.milvus.client.exception;

import io.milvus.client.MilvusClient;

public class UnsupportedServerVersion extends MilvusException {
  private String expect;
  private String actual;

  public UnsupportedServerVersion(String host, String expect, String actual) {
    super(host, false);
    this.expect = expect;
    this.actual = actual;
  }

  @Override
  public String getErrorMessage() {
    return String.format(
        "Milvus client %s is expected to work with Milvus server %s, but the version of the connected server is %s",
        MilvusClient.clientVersion, expect, actual);
  }
}
