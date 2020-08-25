package io.milvus.client;

import java.nio.ByteBuffer;
import java.util.List;

public class VectorEntities {
  private final List<List<Float>> floatVectors;
  private final List<ByteBuffer> binaryVectors;

  public VectorEntities(List<List<Float>> floatVectors, List<ByteBuffer> binaryVectors) {
    this.floatVectors = floatVectors;
    this.binaryVectors = binaryVectors;
  }

  public List<List<Float>> getFloatVectors() {
    return floatVectors;
  }

  public List<ByteBuffer> getBinaryVectors() {
    return binaryVectors;
  }
}
