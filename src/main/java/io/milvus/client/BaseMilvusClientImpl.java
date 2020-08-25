/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package io.milvus.client;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.protobuf.ByteString;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.MethodDescriptor;
import io.grpc.StatusRuntimeException;
import io.milvus.client.exception.MilvusException;
import io.milvus.client.exception.MilvusGrpcException;
import io.milvus.client.exception.ServerSideMilvusException;
import io.milvus.client.exception.UnsupportedServerVersion;
import io.milvus.grpc.*;

import javax.annotation.Nonnull;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/** Actual implementation of interface <code>MilvusClient</code> */
class BaseMilvusClientImpl extends AbstractBaseMilvusClient {
  private static final String SUPPORTED_SERVER_VERSION = "0.10";

  private final ConnectParam connectParam;
  private final ManagedChannel channel;
  private final MilvusServiceGrpc.MilvusServiceBlockingStub blockingStub;
  private final MilvusServiceGrpc.MilvusServiceFutureStub futureStub;

  BaseMilvusClientImpl(ConnectParam connectParam) {
    this.connectParam = connectParam;
    channel = ManagedChannelBuilder.forAddress(connectParam.getHost(), connectParam.getPort())
        .usePlaintext()
        .maxInboundMessageSize(Integer.MAX_VALUE)
        .keepAliveTime(connectParam.getKeepAliveTime(TimeUnit.NANOSECONDS), TimeUnit.NANOSECONDS)
        .keepAliveTimeout(connectParam.getKeepAliveTimeout(TimeUnit.NANOSECONDS), TimeUnit.NANOSECONDS)
        .keepAliveWithoutCalls(connectParam.isKeepAliveWithoutCalls())
        .idleTimeout(connectParam.getIdleTimeout(TimeUnit.NANOSECONDS), TimeUnit.NANOSECONDS)
        .intercept()
        .build();
    blockingStub = MilvusServiceGrpc.newBlockingStub(channel);
    futureStub = MilvusServiceGrpc.newFutureStub(channel);
    try {
      String serverVersion = getServerVersion();
      if (!serverVersion.matches("^" + SUPPORTED_SERVER_VERSION + "(\\..*)?$")) {
        throw new UnsupportedServerVersion(connectParam.getHost(), SUPPORTED_SERVER_VERSION, serverVersion);
      }
    } catch (Throwable t) {
      channel.shutdownNow();
      throw t;
    }
  }

  @Override
  public String host() {
    return connectParam.getHost();
  }

  @Override
  public ConnectivityState getGrpcState() {
    return channel.getState(false);
  }

  @Override
  public void close(long maxWaitSeconds) {
    channel.shutdown();
    long now = System.nanoTime();
    long deadline = now + TimeUnit.SECONDS.toNanos(maxWaitSeconds);
    while (now < deadline && !channel.isTerminated()) {
      try {
        channel.awaitTermination(deadline - now, TimeUnit.NANOSECONDS);
      } catch (InterruptedException ex) {
        // Ignore the interruption
      }
    }
    if (!channel.isTerminated()) {
      channel.shutdownNow();
    }
  }

  @Override
  protected MilvusServiceGrpc.MilvusServiceBlockingStub blockingStub() {
    return blockingStub;
  }

  @Override
  protected MilvusServiceGrpc.MilvusServiceFutureStub futureStub() {
    return futureStub;
  }

  public BaseMilvusClient withTimeout(long timeout, TimeUnit unit) {
    final long timeoutMillis = unit.toMillis(timeout);
    final TimeoutInterceptor interceptor = new TimeoutInterceptor(timeoutMillis);
    final MilvusServiceGrpc.MilvusServiceBlockingStub blockingStub = this.blockingStub.withInterceptors(interceptor);
    final MilvusServiceGrpc.MilvusServiceFutureStub futureStub = this.futureStub.withInterceptors(interceptor);

    return new AbstractBaseMilvusClient() {
      @Override
      public BaseMilvusClient withTimeout(long timeout, TimeUnit timeoutUnit) {
        return BaseMilvusClientImpl.this.withTimeout(timeout, timeoutUnit);
      }

      @Override
      public String host() {
        return BaseMilvusClientImpl.this.host();
      }

      @Override
      public ConnectivityState getGrpcState() {
        return BaseMilvusClientImpl.this.getGrpcState();
      }

      @Override
      public void close(long maxWaitSeconds) {
        BaseMilvusClientImpl.this.close(maxWaitSeconds);
      }

      @Override
      protected MilvusServiceGrpc.MilvusServiceBlockingStub blockingStub() {
        return blockingStub;
      }

      @Override
      protected MilvusServiceGrpc.MilvusServiceFutureStub futureStub() {
        return futureStub;
      }
    };
  }

  private static class TimeoutInterceptor implements ClientInterceptor {
    private final long timeoutMillis;

    TimeoutInterceptor(long timeoutMillis) {
      this.timeoutMillis = timeoutMillis;
    }

    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
        MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
      return next.newCall(method, callOptions.withDeadlineAfter(timeoutMillis, TimeUnit.MILLISECONDS));
    }
  }
}

abstract class AbstractBaseMilvusClient implements BaseMilvusClient {
  private final String extraParamKey = "params";

  protected abstract MilvusServiceGrpc.MilvusServiceBlockingStub blockingStub();

  protected abstract MilvusServiceGrpc.MilvusServiceFutureStub futureStub();

  private Void checkResponseStatus(Status status) {
    if (status.getErrorCode() != ErrorCode.SUCCESS) {
      throw new ServerSideMilvusException(host(), status);
    }
    return null;
  }

  private void translateExceptions(Runnable body) {
    translateExceptions(() -> {
      body.run();
      return null;
    });
  }

  @SuppressWarnings("unchecked")
  private <T> T translateExceptions(Supplier<T> body) {
    try {
      T result = body.get();
      if (result instanceof ListenableFuture) {
        ListenableFuture futureResult = (ListenableFuture) result;
        result = (T) Futures.catching(
            futureResult, Throwable.class, this::translate, MoreExecutors.directExecutor());
      }
      return result;
    } catch (Throwable e) {
      return translate(e);
    }
  }

  private <R> R translate(Throwable e) {
    if (e instanceof MilvusException) {
      throw (MilvusException) e;
    } else if (e instanceof StatusRuntimeException) {
      throw new MilvusGrpcException(host(), (StatusRuntimeException) e);
    } else {
      throw new MilvusException(host(), e);
    }
  }

  @Override
  public void createCollection(@Nonnull CollectionMapping collectionMapping) {
    translateExceptions(() -> {
      CollectionSchema request = CollectionSchema.newBuilder()
          .setCollectionName(collectionMapping.getCollectionName())
          .setDimension(collectionMapping.getDimension())
          .setIndexFileSize(collectionMapping.getIndexFileSize())
          .setMetricType(collectionMapping.getMetricType().getVal())
          .build();
      Status response = blockingStub().createCollection(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public boolean hasCollection(@Nonnull String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      BoolReply response = blockingStub().hasCollection(request);
      checkResponseStatus(response.getStatus());
      return response.getBoolReply();
    });
  }

  @Override
  public void dropCollection(@Nonnull String collectionName) {
    translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      Status response = blockingStub().dropCollection(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public void createIndex(@Nonnull Index index) {
    translateExceptions(() -> {
      IndexParam request = buildIndexParam(index);
      Status response = blockingStub().createIndex(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public ListenableFuture<Void> createIndexAsync(@Nonnull Index index) {
    return translateExceptions(() -> {
      IndexParam request = buildIndexParam(index);
      ListenableFuture<Status> response = futureStub().createIndex(request);
      return Futures.transform(response, this::checkResponseStatus, MoreExecutors.directExecutor());
    });
  }

  @Override
  public void createPartition(String collectionName, String tag) {
    translateExceptions(() -> {
      PartitionParam request = PartitionParam.newBuilder()
          .setCollectionName(collectionName)
          .setTag(tag)
          .build();
      Status response = blockingStub().createPartition(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public boolean hasPartition(String collectionName, String tag) {
    return translateExceptions(() -> {
      PartitionParam request = PartitionParam.newBuilder()
          .setCollectionName(collectionName)
          .setTag(tag)
          .build();
      BoolReply response = blockingStub().hasPartition(request);
      checkResponseStatus(response.getStatus());
      return response.getBoolReply();
    });
  }

  @Override
  public List<String> listPartitions(String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      PartitionList response = blockingStub().showPartitions(request);
      checkResponseStatus(response.getStatus());
      return response.getPartitionTagArrayList();
    });
  }

  @Override
  public void dropPartition(String collectionName, String tag) {
    translateExceptions(() -> {
      PartitionParam request = PartitionParam.newBuilder()
          .setCollectionName(collectionName)
          .setTag(tag)
          .build();
      Status response = blockingStub().dropPartition(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public List<Long> insert(@Nonnull InsertParam insertParam) {
    return translateExceptions(() -> {
      List<RowRecord> rowRecordList =
          buildRowRecordList(insertParam.getFloatVectors(), insertParam.getBinaryVectors());
      io.milvus.grpc.InsertParam request =
          io.milvus.grpc.InsertParam.newBuilder()
              .setCollectionName(insertParam.getCollectionName())
              .addAllRowRecordArray(rowRecordList)
              .addAllRowIdArray(insertParam.getVectorIds())
              .setPartitionTag(insertParam.getPartitionTag())
              .build();
      VectorIds response = blockingStub().insert(request);
      checkResponseStatus(response.getStatus());
      return response.getVectorIdArrayList();
    });
  }

  @Override
  public ListenableFuture<List<Long>> insertAsync(@Nonnull InsertParam insertParam) {
    return translateExceptions(() -> {
      List<RowRecord> rowRecordList =
          buildRowRecordList(insertParam.getFloatVectors(), insertParam.getBinaryVectors());
      io.milvus.grpc.InsertParam request =
          io.milvus.grpc.InsertParam.newBuilder()
              .setCollectionName(insertParam.getCollectionName())
              .addAllRowRecordArray(rowRecordList)
              .addAllRowIdArray(insertParam.getVectorIds())
              .setPartitionTag(insertParam.getPartitionTag())
              .build();
      ListenableFuture<VectorIds> response = futureStub().insert(request);
      return Futures.transform(response, vectorIds -> {
        checkResponseStatus(vectorIds.getStatus());
        return vectorIds.getVectorIdArrayList();
      }, MoreExecutors.directExecutor());
    });
  }

  @Override
  public SearchResult search(@Nonnull SearchParam searchParam) {
    return translateExceptions(() -> {
      List<RowRecord> rowRecordList =
          buildRowRecordList(searchParam.getFloatVectors(), searchParam.getBinaryVectors());
      KeyValuePair extraParam =
          KeyValuePair.newBuilder()
              .setKey(extraParamKey)
              .setValue(searchParam.getParamsInJson())
              .build();
      io.milvus.grpc.SearchParam request =
          io.milvus.grpc.SearchParam.newBuilder()
              .setCollectionName(searchParam.getCollectionName())
              .addAllQueryRecordArray(rowRecordList)
              .addAllPartitionTagArray(searchParam.getPartitionTags())
              .setTopk(searchParam.getTopK())
              .addExtraParams(extraParam)
              .build();
      TopKQueryResult response = blockingStub().search(request);
      checkResponseStatus(response.getStatus());
      return buildSearchResult(response);
    });
  }

  @Override
  public ListenableFuture<SearchResult> searchAsync(@Nonnull SearchParam searchParam) {
    return translateExceptions(() -> {
      List<RowRecord> rowRecordList =
          buildRowRecordList(searchParam.getFloatVectors(), searchParam.getBinaryVectors());

      KeyValuePair extraParam =
          KeyValuePair.newBuilder()
              .setKey(extraParamKey)
              .setValue(searchParam.getParamsInJson())
              .build();

      io.milvus.grpc.SearchParam request =
          io.milvus.grpc.SearchParam.newBuilder()
              .setCollectionName(searchParam.getCollectionName())
              .addAllQueryRecordArray(rowRecordList)
              .addAllPartitionTagArray(searchParam.getPartitionTags())
              .setTopk(searchParam.getTopK())
              .addExtraParams(extraParam)
              .build();

      ListenableFuture<TopKQueryResult> response = futureStub().search(request);
      return Futures.transform(response, result -> {
        checkResponseStatus(result.getStatus());
        return buildSearchResult(result);
      }, MoreExecutors.directExecutor());
    });
  }

  @Override
  public CollectionMapping getCollectionInfo(@Nonnull String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      CollectionSchema response = blockingStub().describeCollection(request);
      checkResponseStatus(response.getStatus());
      return new CollectionMapping.Builder(response.getCollectionName(), response.getDimension())
          .withIndexFileSize(response.getIndexFileSize())
          .withMetricType(MetricType.valueOf(response.getMetricType()))
          .build();
    });
  }

  @Override
  public List<String> listCollections() {
    return translateExceptions(() -> {
      Command request = Command.newBuilder().setCmd("").build();
      CollectionNameList response = blockingStub().showCollections(request);
      checkResponseStatus(response.getStatus());
      return response.getCollectionNamesList();
    });
  }

  @Override
  public long countEntities(@Nonnull String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      CollectionRowCount response = blockingStub().countCollection(request);
      checkResponseStatus(response.getStatus());
      return response.getCollectionRowCount();
    });
  }

  @Override
  public String getServerStatus() {
    return command("status");
  }

  @Override
  public String getServerVersion() {
    return command("version");
  }

  @Override
  public String command(@Nonnull String command) {
    return translateExceptions(() -> {
      Command request = Command.newBuilder().setCmd(command).build();
      StringReply response = blockingStub().cmd(request);
      checkResponseStatus(response.getStatus());
      return response.getStringReply();
    });
  }

  @Override
  public void loadCollection(@Nonnull String collectionName) {
    translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      Status response = blockingStub().preloadCollection(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public Index getIndexInfo(@Nonnull String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      IndexParam response = blockingStub().describeIndex(request);
      checkResponseStatus(response.getStatus());
      String extraParam = "";
      for (KeyValuePair kv : response.getExtraParamsList()) {
        if (kv.getKey().contentEquals(extraParamKey)) {
          extraParam = kv.getValue();
        }
      }
      return new Index.Builder(response.getCollectionName(), IndexType.valueOf(response.getIndexType()))
          .withParamsInJson(extraParam)
          .build();
    });
  }

  @Override
  public void dropIndex(@Nonnull String collectionName) {
    translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      Status response = blockingStub().dropIndex(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public String getCollectionStats(String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      CollectionInfo response = blockingStub().showCollectionInfo(request);
      checkResponseStatus(response.getStatus());
      return response.getJsonInfo();
    });
  }

  @Override
  public VectorEntities getEntityByID(String collectionName, List<Long> ids) {
    return translateExceptions(() -> {
      VectorsIdentity request = VectorsIdentity.newBuilder()
          .setCollectionName(collectionName)
          .addAllIdArray(ids)
          .build();
      VectorsData response = blockingStub().getVectorsByID(request);
      checkResponseStatus(response.getStatus());
      List<List<Float>> floatVectors = new ArrayList<>(ids.size());
      List<ByteBuffer> binaryVectors = new ArrayList<>(ids.size());
      for (int i = 0; i < ids.size(); i++) {
        floatVectors.add(response.getVectorsData(i).getFloatDataList());
        binaryVectors.add(response.getVectorsData(i).getBinaryData().asReadOnlyByteBuffer());
      }
      return new VectorEntities(floatVectors, binaryVectors);
    });
  }

  @Override
  public List<Long> listIDInSegment(String collectionName, String segmentName) {
    return translateExceptions(() -> {
      GetVectorIDsParam request =
          GetVectorIDsParam.newBuilder()
              .setCollectionName(collectionName)
              .setSegmentName(segmentName)
              .build();
      VectorIds response = blockingStub().getVectorIDs(request);
      return response.getVectorIdArrayList();
    });
  }

  @Override
  public void deleteEntityByID(String collectionName, List<Long> ids) {
    translateExceptions(() -> {
      DeleteByIDParam request =
          DeleteByIDParam.newBuilder().setCollectionName(collectionName).addAllIdArray(ids).build();
      Status response = blockingStub().deleteByID(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public void flush(List<String> collectionNames) {
    translateExceptions(() -> {
      FlushParam request = FlushParam.newBuilder().addAllCollectionNameArray(collectionNames).build();
      Status response = blockingStub().flush(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public ListenableFuture<Void> flushAsync(@Nonnull List<String> collectionNames) {
    return translateExceptions(() -> {
      FlushParam request = FlushParam.newBuilder().addAllCollectionNameArray(collectionNames).build();
      ListenableFuture<Status> response = futureStub().flush(request);
      return Futures.transform(response, this::checkResponseStatus, MoreExecutors.directExecutor());
    });
  }

  @Override
  public void flush(String collectionName) {
    flush(ImmutableList.of(collectionName));
  }

  @Override
  public ListenableFuture<Void> flushAsync(String collectionName) {
    return translateExceptions(() -> {
      return flushAsync(ImmutableList.of(collectionName));
    });
  }

  @Override
  public void compact(String collectionName) {
    translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      Status response = blockingStub().compact(request);
      checkResponseStatus(response);
    });
  }

  @Override
  public ListenableFuture<Void> compactAsync(@Nonnull String collectionName) {
    return translateExceptions(() -> {
      CollectionName request = CollectionName.newBuilder().setCollectionName(collectionName).build();
      ListenableFuture<Status> response = futureStub().compact(request);
      return Futures.transform(response, this::checkResponseStatus, MoreExecutors.directExecutor());
    });
  }

  ///////////////////// Util Functions/////////////////////
  private IndexParam buildIndexParam(Index index) {
    KeyValuePair extraParam = KeyValuePair.newBuilder()
        .setKey(extraParamKey)
        .setValue(index.getParamsInJson())
        .build();
    return IndexParam.newBuilder()
        .setCollectionName(index.getCollectionName())
        .setIndexType(index.getIndexType().getVal())
        .addExtraParams(extraParam)
        .build();
  }

  private List<RowRecord> buildRowRecordList(
      @Nonnull List<List<Float>> floatVectors, @Nonnull List<ByteBuffer> binaryVectors) {
    int largerSize = Math.max(floatVectors.size(), binaryVectors.size());

    List<RowRecord> rowRecordList = new ArrayList<>(largerSize);

    for (int i = 0; i < largerSize; ++i) {

      RowRecord.Builder rowRecordBuilder = RowRecord.newBuilder();

      if (i < floatVectors.size()) {
        rowRecordBuilder.addAllFloatData(floatVectors.get(i));
      }
      if (i < binaryVectors.size()) {
        ((Buffer) binaryVectors.get(i)).rewind();
        rowRecordBuilder.setBinaryData(ByteString.copyFrom(binaryVectors.get(i)));
      }

      rowRecordList.add(rowRecordBuilder.build());
    }

    return rowRecordList;
  }

  private SearchResult buildSearchResult(TopKQueryResult topKQueryResult) {
    final int numQueries = (int) topKQueryResult.getRowNum();
    final int topK = numQueries == 0
        ? 0
        : topKQueryResult.getIdsCount() / numQueries; // Guaranteed to be divisible from server side

    List<List<Long>> resultIdsList = new ArrayList<>(numQueries);
    List<List<Float>> resultDistancesList = new ArrayList<>(numQueries);

    if (topK > 0) {
      for (int i = 0; i < numQueries; i++) {
        // Process result of query i
        int pos = i * topK;
        while (pos < i * topK + topK && topKQueryResult.getIdsList().get(pos) != -1) {
          pos++;
        }
        resultIdsList.add(topKQueryResult.getIdsList().subList(i * topK, pos));
        resultDistancesList.add(topKQueryResult.getDistancesList().subList(i * topK, pos));
      }
    }

    return new SearchResult(numQueries, topK, resultIdsList, resultDistancesList);
  }
}
