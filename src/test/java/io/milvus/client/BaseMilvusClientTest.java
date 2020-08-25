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
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.Status;
import io.milvus.client.exception.MilvusGrpcException;
import io.milvus.client.exception.ServerSideMilvusException;
import io.milvus.client.exception.UnsupportedServerVersion;
import io.milvus.grpc.ErrorCode;
import org.apache.commons.text.RandomStringGenerator;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;

@Testcontainers
class BaseMilvusClientTest {

  private BaseMilvusClient client;

  private RandomStringGenerator generator;

  private String randomCollectionName;
  private int size;
  private int dimension;

  @Container
  private GenericContainer milvusContainer =
      new GenericContainer("milvusdb/milvus:0.10.1-cpu-d072020-bd02b1")
          .withExposedPorts(19530);

  // Helper function that generates random float vectors
  static List<List<Float>> generateFloatVectors(int vectorCount, int dimension) {
    SplittableRandom splittableRandom = new SplittableRandom();
    List<List<Float>> vectors = new ArrayList<>(vectorCount);
    for (int i = 0; i < vectorCount; ++i) {
      splittableRandom = splittableRandom.split();
      DoubleStream doubleStream = splittableRandom.doubles(dimension);
      List<Float> vector =
          doubleStream.boxed().map(Double::floatValue).collect(Collectors.toList());
      vectors.add(vector);
    }
    return vectors;
  }

  // Helper function that generates random binary vectors
  static List<ByteBuffer> generateBinaryVectors(int vectorCount, int dimension) {
    Random random = new Random();
    List<ByteBuffer> vectors = new ArrayList<>(vectorCount);
    final int dimensionInByte = dimension / 8;
    for (int i = 0; i < vectorCount; ++i) {
      ByteBuffer byteBuffer = ByteBuffer.allocate(dimensionInByte);
      random.nextBytes(byteBuffer.array());
      vectors.add(byteBuffer);
    }
    return vectors;
  }

  // Helper function that normalizes a vector if you are using IP (Inner Product) as your metric
  // type
  static List<Float> normalizeVector(List<Float> vector) {
    float squareSum = vector.stream().map(x -> x * x).reduce((float) 0, Float::sum);
    final float norm = (float) Math.sqrt(squareSum);
    vector = vector.stream().map(x -> x / norm).collect(Collectors.toList());
    return vector;
  }

  ConnectParam.Builder connectParamBuilder() {
    return connectParamBuilder(milvusContainer);
  }

  ConnectParam.Builder connectParamBuilder(GenericContainer milvusContainer) {
    return connectParamBuilder(milvusContainer.getHost(), milvusContainer.getFirstMappedPort());
  }

  ConnectParam.Builder connectParamBuilder(String host, int port) {
    return new ConnectParam.Builder().withHost(host).withPort(port);
  }

  @BeforeEach
  void setUp() {
    ConnectParam connectParam = connectParamBuilder().build();
    client = BaseMilvusClient.create(connectParam);

    generator = new RandomStringGenerator.Builder().withinRange('a', 'z').build();
    randomCollectionName = generator.generate(10);
    size = 100000;
    dimension = 128;
    CollectionMapping collectionMapping =
        new CollectionMapping.Builder(randomCollectionName, dimension)
            .withIndexFileSize(1024)
            .withMetricType(MetricType.IP)
            .build();

    client.createCollection(collectionMapping);
  }

  @AfterEach
  void tearDown() {
    client.close();
  }

  @Test
  void idleTest() throws InterruptedException {
    ConnectParam connectParam = connectParamBuilder()
        .withIdleTimeout(1, TimeUnit.SECONDS)
        .build();
    client = BaseMilvusClient.create(connectParam);
    TimeUnit.SECONDS.sleep(2);
    // A new RPC would take the channel out of idle mode
    assertEquals(ImmutableList.of(randomCollectionName), client.listCollections());
  }

  @Test
  void setInvalidConnectParam() {
    assertThrows(
        IllegalArgumentException.class,
        () -> connectParamBuilder().withPort(66666).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> connectParamBuilder().withConnectTimeout(-1, TimeUnit.MILLISECONDS).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> connectParamBuilder().withKeepAliveTime(-1, TimeUnit.MILLISECONDS).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> connectParamBuilder().withKeepAliveTimeout(-1, TimeUnit.MILLISECONDS).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> connectParamBuilder().withIdleTimeout(-1, TimeUnit.MILLISECONDS).build());
  }

  @Test
  void connectUnreachableHost() {
    int port = milvusContainer.getFirstMappedPort();
    milvusContainer.stop();

    MilvusGrpcException e = assertThrows(MilvusGrpcException.class, () -> client.getServerVersion());
    assertEquals(Status.Code.UNAVAILABLE, e.getCause().getStatus().getCode());

    e = assertThrows(MilvusGrpcException.class, () ->
        MilvusClientFactory.createBaseMilvusClient(connectParamBuilder(milvusContainer.getHost(), port).build()));
    assertEquals(Status.Code.UNAVAILABLE, e.getCause().getStatus().getCode());
  }

  @Test
  void unsupportedServerVersion() {
    GenericContainer unsupportedMilvus =
        new GenericContainer("milvusdb/milvus:0.9.1-cpu-d052920-e04ed5")
            .withExposedPorts(19530);
    try {
      unsupportedMilvus.start();
      ConnectParam connectParam = connectParamBuilder(unsupportedMilvus).build();
      assertThrows(UnsupportedServerVersion.class, () -> MilvusClientFactory.createBaseMilvusClient(connectParam));
    } finally {
      unsupportedMilvus.stop();
    }
  }

  @Test
  void grpcTimeout() {
    insert();
    BaseMilvusClient timeoutClient = client.withTimeout(1, TimeUnit.MILLISECONDS);
    MilvusGrpcException e = assertThrows(
        MilvusGrpcException.class,
        () -> timeoutClient.createIndex(
            new Index.Builder(randomCollectionName, IndexType.FLAT)
                .withParamsInJson("{\"nlist\": 16384}").build()));
    assertEquals(Status.Code.DEADLINE_EXCEEDED, e.getCause().getStatus().getCode());
  }

  @Test
  void createInvalidCollection() {
    String invalidCollectionName = "╯°□°）╯";
    CollectionMapping invalidCollectionMapping =
        new CollectionMapping.Builder(invalidCollectionName, dimension).build();
    ServerSideMilvusException e = assertThrows(
        ServerSideMilvusException.class,
        () -> client.createCollection(invalidCollectionMapping));
    assertEquals(ErrorCode.ILLEGAL_COLLECTION_NAME, e.getErrorCode());
  }

  @Test
  void hasCollection() {
    assertTrue(client.hasCollection(randomCollectionName));
  }

  @Test
  void dropCollection() {
    String nonExistingCollectionName = generator.generate(10);
    ServerSideMilvusException e = assertThrows(
        ServerSideMilvusException.class,
        () -> client.dropCollection(nonExistingCollectionName));
    assertEquals(ErrorCode.COLLECTION_NOT_EXISTS, e.getErrorCode());

    client.dropCollection(randomCollectionName);
    assertFalse(client.hasCollection(randomCollectionName));
  }

  @Test
  void partitionTest() {
    final String tag1 = "tag1";
    client.createPartition(randomCollectionName, tag1);

    final String tag2 = "tag2";
    client.createPartition(randomCollectionName, tag2);

    List<String> partitions = client.listPartitions(randomCollectionName);
    assertEquals(3, partitions.size()); // two tags plus _default

    List<List<Float>> vectors1 = generateFloatVectors(size, dimension);
    List<Long> vectorIds1 = LongStream.range(0, size).boxed().collect(Collectors.toList());
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName)
            .withFloatVectors(vectors1)
            .withVectorIds(vectorIds1)
            .withPartitionTag(tag1)
            .build();
    List<Long> ids1 = client.insert(insertParam);
    assertEquals(vectorIds1, ids1);

    List<List<Float>> vectors2 = generateFloatVectors(size, dimension);
    List<Long> vectorIds2 = LongStream.range(size, size * 2).boxed().collect(Collectors.toList());
    insertParam =
        new InsertParam.Builder(randomCollectionName)
            .withFloatVectors(vectors2)
            .withVectorIds(vectorIds2)
            .withPartitionTag(tag2)
            .build();
    List<Long> ids2 = client.insert(insertParam);
    assertEquals(vectorIds2, ids2);

    client.flush(randomCollectionName);

    assertEquals(size * 2, client.countEntities(randomCollectionName));

    final int searchSize = 1;
    final long topK = 10;

    List<List<Float>> vectorsToSearch1 = vectors1.subList(0, searchSize);
    List<String> partitionTags1 = new ArrayList<>();
    partitionTags1.add(tag1);
    SearchParam searchParam1 =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch1)
            .withTopK(topK)
            .withPartitionTags(partitionTags1)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResult searchResult1 = client.search(searchParam1);
    List<List<Long>> resultIdsList1 = searchResult1.getResultIdsList();
    assertEquals(searchSize, resultIdsList1.size());
    assertTrue(vectorIds1.containsAll(resultIdsList1.get(0)));

    List<List<Float>> vectorsToSearch2 = vectors2.subList(0, searchSize);
    List<String> partitionTags2 = new ArrayList<>();
    partitionTags2.add(tag2);
    SearchParam searchParam2 =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch2)
            .withTopK(topK)
            .withPartitionTags(partitionTags2)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResult searchResult2 = client.search(searchParam2);
    List<List<Long>> resultIdsList2 = searchResult2.getResultIdsList();
    assertEquals(searchSize, resultIdsList2.size());
    assertTrue(vectorIds2.containsAll(resultIdsList2.get(0)));

    assertTrue(Collections.disjoint(resultIdsList1, resultIdsList2));

    assertTrue(client.hasPartition(randomCollectionName, tag1));
    client.dropPartition(randomCollectionName, tag1);
    assertFalse(client.hasPartition(randomCollectionName, tag1));

    client.dropPartition(randomCollectionName, tag2);
  }

  @Test
  void createIndex() {
    insert();
    client.flush(randomCollectionName);

    Index index = new Index.Builder(randomCollectionName, IndexType.IVF_SQ8)
        .withParamsInJson("{\"nlist\": 16384}")
        .build();

    client.createIndex(index);
  }

  @Test
  void createIndexAsync() throws ExecutionException, InterruptedException {
    insert();
    client.flush(randomCollectionName);

    Index index = new Index.Builder(randomCollectionName, IndexType.IVF_SQ8)
        .withParamsInJson("{\"nlist\": 16384}")
        .build();

    client.createIndexAsync(index).get();
  }

  @Test
  void insert() {
    assertEquals(size, insertData().size());
  }

  private List<Long> insertData() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam = new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    return client.insert(insertParam);
  }

  @Test
  void insertAsync() throws ExecutionException, InterruptedException {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam = new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    List<Long> vectorIds = client.insertAsync(insertParam).get();
    assertEquals(size, vectorIds.size());
  }

  @Test
  void insertBinary() {
    final int binaryDimension = 10000;

    String binaryCollectionName = generator.generate(10);
    CollectionMapping collectionMapping =
        new CollectionMapping.Builder(binaryCollectionName, binaryDimension)
            .withIndexFileSize(1024)
            .withMetricType(MetricType.JACCARD)
            .build();

    client.createCollection(collectionMapping);

    List<ByteBuffer> vectors = generateBinaryVectors(size, binaryDimension);
    InsertParam insertParam = new InsertParam.Builder(binaryCollectionName).withBinaryVectors(vectors).build();
    List<Long> vectorIds = client.insert(insertParam);
    assertEquals(size, vectorIds.size());
  }

  @Test
  void search() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    vectors = vectors.stream().map(BaseMilvusClientTest::normalizeVector).collect(Collectors.toList());
    InsertParam insertParam = new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    List<Long> vectorIds = client.insert(insertParam);
    assertEquals(size, vectorIds.size());

    client.flush(randomCollectionName);

    final int searchSize = 5;
    List<List<Float>> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResult searchResult = client.search(searchParam);
    List<List<Long>> resultIdsList = searchResult.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResult.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResult.QueryResult>> queryResultsList = searchResult.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());

    final double epsilon = 0.001;
    for (int i = 0; i < searchSize; i++) {
      SearchResult.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
      assertEquals(vectorIds.get(i), firstQueryResult.getVectorId());
      assertEquals(vectorIds.get(i), resultIdsList.get(i).get(0));
      assertTrue(Math.abs(1 - firstQueryResult.getDistance()) < epsilon);
      assertTrue(Math.abs(1 - resultDistancesList.get(i).get(0)) < epsilon);
    }
  }

  @Test
  void searchAsync() throws ExecutionException, InterruptedException {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    vectors = vectors.stream().map(BaseMilvusClientTest::normalizeVector).collect(Collectors.toList());
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    List<Long> vectorIds = client.insert(insertParam);
    assertEquals(size, vectorIds.size());

    client.flush(randomCollectionName);

    final int searchSize = 5;
    List<List<Float>> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    ListenableFuture<SearchResult> searchResultFuture = client.searchAsync(searchParam);
    SearchResult searchResult = searchResultFuture.get();
    List<List<Long>> resultIdsList = searchResult.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResult.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResult.QueryResult>> queryResultsList = searchResult.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());
    final double epsilon = 0.001;
    for (int i = 0; i < searchSize; i++) {
      SearchResult.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
      assertEquals(vectorIds.get(i), firstQueryResult.getVectorId());
      assertEquals(vectorIds.get(i), resultIdsList.get(i).get(0));
      assertTrue(Math.abs(1 - firstQueryResult.getDistance()) < epsilon);
      assertTrue(Math.abs(1 - resultDistancesList.get(i).get(0)) < epsilon);
    }
  }

  @Test
  void searchBinary() {
    final int binaryDimension = 10000;

    String binaryCollectionName = generator.generate(10);
    CollectionMapping collectionMapping =
        new CollectionMapping.Builder(binaryCollectionName, binaryDimension)
            .withIndexFileSize(1024)
            .withMetricType(MetricType.JACCARD)
            .build();

    client.createCollection(collectionMapping);

    List<ByteBuffer> vectors = generateBinaryVectors(size, binaryDimension);
    InsertParam insertParam =
        new InsertParam.Builder(binaryCollectionName).withBinaryVectors(vectors).build();
    List<Long> vectorIds = client.insert(insertParam);
    assertEquals(size, vectorIds.size());

    client.flush(binaryCollectionName);

    final int searchSize = 5;
    List<ByteBuffer> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(binaryCollectionName)
            .withBinaryVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResult searchResult = client.search(searchParam);
    List<List<Long>> resultIdsList = searchResult.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResult.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResult.QueryResult>> queryResultsList = searchResult.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());
    for (int i = 0; i < searchSize; i++) {
      SearchResult.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
      assertEquals(vectorIds.get(i), firstQueryResult.getVectorId());
      assertEquals(vectorIds.get(i), resultIdsList.get(i).get(0));
    }
  }

  @Test
  void getCollectionInfo() {
    CollectionMapping collectionMapping = client.getCollectionInfo(randomCollectionName);
    assertEquals(randomCollectionName, collectionMapping.getCollectionName());

    String nonExistingCollectionName = generator.generate(10);
    ServerSideMilvusException e = assertThrows(
        ServerSideMilvusException.class,
        () -> client.getCollectionInfo(nonExistingCollectionName));
    assertEquals(ErrorCode.COLLECTION_NOT_EXISTS, e.getErrorCode());
  }

  @Test
  void listCollections() {
    List<String> collectionNames = client.listCollections();
    assertTrue(collectionNames.contains(randomCollectionName));
  }

  @Test
  void serverStatus() {
    assertEquals("OK", client.getServerStatus());
  }

  @Test
  void serverVersion() {
    assertEquals("0.10.1", client.getServerVersion());
  }

  @Test
  void countEntities() {
    insert();
    client.flush(randomCollectionName);
    assertEquals(size, client.countEntities(randomCollectionName));
  }

  @Test
  void loadCollection() {
    insert();
    client.flush(randomCollectionName);
    client.loadCollection(randomCollectionName);
  }

  @Test
  void getIndexInfo() {
    String nonExistingCollectionName = generator.generate(10);
    ServerSideMilvusException e = assertThrows(
        ServerSideMilvusException.class,
        () -> client.getIndexInfo(nonExistingCollectionName));
    assertEquals(ErrorCode.COLLECTION_NOT_EXISTS, e.getErrorCode());

    createIndex();
    Index index = client.getIndexInfo(randomCollectionName);
    assertEquals(index.getCollectionName(), randomCollectionName);
    assertEquals(index.getIndexType(), IndexType.IVF_SQ8);
  }

  @Test
  void dropIndex() {
    client.dropIndex(randomCollectionName);
  }

  @Test
  void getCollectionStats() {
    insert();
    client.flush(randomCollectionName);

    String jsonString = client.getCollectionStats(randomCollectionName);

    JSONObject jsonInfo = new JSONObject(jsonString);
    assertTrue(jsonInfo.getInt("row_count") == size);

    JSONArray partitions = jsonInfo.getJSONArray("partitions");
    JSONObject partitionInfo = partitions.getJSONObject(0);
    assertEquals(partitionInfo.getString("tag"), "_default");
    assertEquals(partitionInfo.getInt("row_count"), size);

    JSONArray segments = partitionInfo.getJSONArray("segments");
    JSONObject segmentInfo = segments.getJSONObject(0);
    assertEquals(segmentInfo.getString("index_name"), "IDMAP");
    assertEquals(segmentInfo.getInt("row_count"), size);
  }

  @Test
  void getEntityByID() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    List<Long> vectorIds = client.insert(insertParam);
    assertEquals(size, vectorIds.size());

    client.flush(randomCollectionName);

    VectorEntities vectorEntities = client.getEntityByID(randomCollectionName, vectorIds.subList(0, 100));
    ByteBuffer bb = vectorEntities.getBinaryVectors().get(0);
    assertTrue(bb == null || bb.remaining() == 0);

    assertArrayEquals(vectorEntities.getFloatVectors().get(0).toArray(), vectors.get(0).toArray());
  }

  @Test
  void getVectorIds() {
    insertData();
    client.flush(randomCollectionName);

    String collectionStats = client.getCollectionStats(randomCollectionName);
    String segmentName = extractSegmentName(collectionStats);
    List<Long> vectorIds = client.listIDInSegment(randomCollectionName, segmentName);
    assertFalse(vectorIds.isEmpty());
  }

  @Test
  void deleteEntityByID() {
    List<Long> vectorIds = insertData();
    client.flush(randomCollectionName);

    client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, 100));
    client.flush(randomCollectionName);
    assertEquals(size - 100, client.countEntities(randomCollectionName));
  }

  @Test
  void compact() {
    List<Long> vectorIds = insertData();
    client.flush(randomCollectionName);

    String collectionStats = client.getCollectionStats(randomCollectionName);
    long previousSegmentSize = extractSegmentSize(collectionStats);

    client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, (int) size / 2));
    client.flush(randomCollectionName);
    client.compact(randomCollectionName);

    collectionStats = client.getCollectionStats(randomCollectionName);
    long currentSegmentSize = extractSegmentSize(collectionStats);
    assertTrue(currentSegmentSize < previousSegmentSize);
  }

  @Test
  void compactAsync() throws ExecutionException, InterruptedException {
    List<Long> vectorIds = insertData();
    client.flush(randomCollectionName);

    String collectionStats = client.getCollectionStats(randomCollectionName);
    long previousSegmentSize = extractSegmentSize(collectionStats);

    client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, (int) size / 2));
    client.flush(randomCollectionName);
    client.compactAsync(randomCollectionName).get();

    collectionStats = client.getCollectionStats(randomCollectionName);
    long currentSegmentSize = extractSegmentSize(collectionStats);

    assertTrue(currentSegmentSize < previousSegmentSize);
  }

  private long extractSegmentSize(String collectionStats) {
    return extractSegmentInfo(collectionStats).getLong("data_size");
  }

  private String extractSegmentName(String collectionStats) {
    return extractSegmentInfo(collectionStats).getString("name");
  }

  private JSONObject extractSegmentInfo(String collectionStats) {
    return new JSONObject(collectionStats)
        .getJSONArray("partitions")
        .getJSONObject(0)
        .getJSONArray("segments")
        .getJSONObject(0);
  }
}
