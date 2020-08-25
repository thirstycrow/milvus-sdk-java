package io.milvus.client;

public class MilvusClientFactory {

  public static BaseMilvusClient createBaseMilvusClient(ConnectParam connectParam) {
    return new BaseMilvusClientImpl(connectParam);
  }

  public static MilvusClient createClassicMilvusClient(ConnectParam connectParam) {
    return new MilvusGrpcClient(createBaseMilvusClient(connectParam));
  }

  private MilvusClientFactory() {
  }
}
