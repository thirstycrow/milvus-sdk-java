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

import com.google.common.util.concurrent.ListenableFuture;
import io.milvus.client.exception.InitializationFailed;
import io.milvus.client.exception.UnsupportedServerVersion;
import org.apache.commons.text.RandomStringGenerator;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.HostPortWaitStrategy;
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
class MilvusClientTest {

  private MilvusClient client;

  private RandomStringGenerator generator;

  private String randomCollectionName;
  private int size;
  private int dimension;

  @Container
  private GenericContainer milvus =
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
    return connectParamBuilder(milvus);
  }

  ConnectParam.Builder connectParamBuilder(GenericContainer milvusContainer) {
    return connectParamBuilder(milvusContainer.getHost(), milvusContainer.getFirstMappedPort());
  }

  ConnectParam.Builder connectParamBuilder(String host, int port) {
    return new ConnectParam.Builder().withHost(host).withPort(port);
  }

  @BeforeEach
  void setUp() throws Exception {
    ConnectParam connectParam = connectParamBuilder().build();
    client = MilvusClient.create(connectParam);

    generator = new RandomStringGenerator.Builder().withinRange('a', 'z').build();
    randomCollectionName = generator.generate(10);
    size = 100000;
    dimension = 128;
    CollectionMapping collectionMapping =
        new CollectionMapping.Builder(randomCollectionName, dimension)
            .withIndexFileSize(1024)
            .withMetricType(MetricType.IP)
            .build();

    assertTrue(client.createCollection(collectionMapping).ok());
  }

  @AfterEach
  void tearDown() throws InterruptedException {
    client.close();
  }

  @Test
  void idleTest() throws InterruptedException {
    ConnectParam connectParam = connectParamBuilder()
        .withIdleTimeout(1, TimeUnit.SECONDS)
        .build();
    client = MilvusClient.create(connectParam);
    TimeUnit.SECONDS.sleep(2);
    // A new RPC would take the channel out of idle mode
    assertTrue(client.listCollections().ok());
  }

  @Test
  void setInvalidConnectParam() {
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          ConnectParam connectParam = connectParamBuilder().withPort(66666).build();
        });
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          ConnectParam connectParam =
              connectParamBuilder().withConnectTimeout(-1, TimeUnit.MILLISECONDS).build();
        });
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          ConnectParam connectParam =
              connectParamBuilder().withKeepAliveTime(-1, TimeUnit.MILLISECONDS).build();
        });
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          ConnectParam connectParam =
              connectParamBuilder().withKeepAliveTimeout(-1, TimeUnit.MILLISECONDS).build();
        });
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          ConnectParam connectParam =
              connectParamBuilder().withIdleTimeout(-1, TimeUnit.MILLISECONDS).build();
        });
  }

  @Test
  void connectUnreachableHost() {
    int port = milvus.getFirstMappedPort();
    milvus.stop();
    assertEquals(Response.Status.RPC_ERROR, client.getServerVersion().getStatus());
    assertEquals(Response.Status.CLIENT_NOT_CONNECTED, client.getServerVersion().getStatus());
    assertThrows(InitializationFailed.class,
        () -> MilvusClient.create(connectParamBuilder(milvus.getHost(), port).build()));
  }

  @Test
  void unsupportedServerVersion() {
    GenericContainer unsupportedMilvus =
        new GenericContainer("milvusdb/milvus:0.9.1-cpu-d052920-e04ed5")
            .withExposedPorts(19530);
    try {
      unsupportedMilvus.start();
      ConnectParam connectParam = connectParamBuilder(unsupportedMilvus).build();
      assertThrows(UnsupportedServerVersion.class, () -> MilvusClient.create(connectParam));
    } finally {
      unsupportedMilvus.stop();
    }
  }

  @Test
  void createInvalidCollection() {
    String invalidCollectionName = "╯°□°）╯";
    CollectionMapping invalidCollectionMapping =
        new CollectionMapping.Builder(invalidCollectionName, dimension).build();
    Response createCollectionResponse = client.createCollection(invalidCollectionMapping);
    assertFalse(createCollectionResponse.ok());
    assertEquals(Response.Status.ILLEGAL_COLLECTION_NAME, createCollectionResponse.getStatus());
  }

  @Test
  void hasCollection() {
    HasCollectionResponse hasCollectionResponse = client.hasCollection(randomCollectionName);
    assertTrue(hasCollectionResponse.ok());
    assertTrue(hasCollectionResponse.hasCollection());
  }

  @Test
  void dropCollection() {
    String nonExistingCollectionName = generator.generate(10);
    Response dropCollectionResponse = client.dropCollection(nonExistingCollectionName);
    assertFalse(dropCollectionResponse.ok());
    assertEquals(Response.Status.COLLECTION_NOT_EXISTS, dropCollectionResponse.getStatus());
  }

  @Test
  void partitionTest() {
    final String tag1 = "tag1";
    Response createPartitionResponse = client.createPartition(randomCollectionName, tag1);
    assertTrue(createPartitionResponse.ok());

    final String tag2 = "tag2";
    createPartitionResponse = client.createPartition(randomCollectionName, tag2);
    assertTrue(createPartitionResponse.ok());

    ListPartitionsResponse listPartitionsResponse = client.listPartitions(randomCollectionName);
    assertTrue(listPartitionsResponse.ok());
    assertEquals(3, listPartitionsResponse.getPartitionList().size()); // two tags plus _default

    List<List<Float>> vectors1 = generateFloatVectors(size, dimension);
    List<Long> vectorIds1 = LongStream.range(0, size).boxed().collect(Collectors.toList());
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName)
            .withFloatVectors(vectors1)
            .withVectorIds(vectorIds1)
            .withPartitionTag(tag1)
            .build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<List<Float>> vectors2 = generateFloatVectors(size, dimension);
    List<Long> vectorIds2 = LongStream.range(size, size * 2).boxed().collect(Collectors.toList());
    insertParam =
        new InsertParam.Builder(randomCollectionName)
            .withFloatVectors(vectors2)
            .withVectorIds(vectorIds2)
            .withPartitionTag(tag2)
            .build();
    insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());

    assertTrue(client.flush(randomCollectionName).ok());

    assertEquals(size * 2, client.countEntities(randomCollectionName).getCollectionEntityCount());

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
    SearchResponse searchResponse1 = client.search(searchParam1);
    assertTrue(searchResponse1.ok());
    List<List<Long>> resultIdsList1 = searchResponse1.getResultIdsList();
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
    SearchResponse searchResponse2 = client.search(searchParam2);
    assertTrue(searchResponse2.ok());
    List<List<Long>> resultIdsList2 = searchResponse2.getResultIdsList();
    assertEquals(searchSize, resultIdsList2.size());
    assertTrue(vectorIds2.containsAll(resultIdsList2.get(0)));

    assertTrue(Collections.disjoint(resultIdsList1, resultIdsList2));

    HasPartitionResponse testHasPartition = client.hasPartition(randomCollectionName, tag1);
    assertTrue(testHasPartition.hasPartition());

    Response dropPartitionResponse = client.dropPartition(randomCollectionName, tag1);
    assertTrue(dropPartitionResponse.ok());

    testHasPartition = client.hasPartition(randomCollectionName, tag1);
    assertFalse(testHasPartition.hasPartition());

    dropPartitionResponse = client.dropPartition(randomCollectionName, tag2);
    assertTrue(dropPartitionResponse.ok());
  }

  @Test
  void createIndex() {
    insert();
    assertTrue(client.flush(randomCollectionName).ok());

    Index index =
        new Index.Builder(randomCollectionName, IndexType.IVF_SQ8)
            .withParamsInJson("{\"nlist\": 16384}")
            .build();

    Response createIndexResponse = client.createIndex(index);
    assertTrue(createIndexResponse.ok());
  }

  @Test
  void createIndexAsync() throws ExecutionException, InterruptedException {
    insert();
    assertTrue(client.flush(randomCollectionName).ok());

    Index index =
        new Index.Builder(randomCollectionName, IndexType.IVF_SQ8)
            .withParamsInJson("{\"nlist\": 16384}")
            .build();

    ListenableFuture<Response> createIndexResponseFuture = client.createIndexAsync(index);
    Response createIndexResponse = createIndexResponseFuture.get();
    assertTrue(createIndexResponse.ok());
  }

  @Test
  void insert() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    assertEquals(size, insertResponse.getVectorIds().size());
  }

  @Test
  void insertAsync() throws ExecutionException, InterruptedException {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    ListenableFuture<InsertResponse> insertResponseFuture = client.insertAsync(insertParam);
    InsertResponse insertResponse = insertResponseFuture.get();
    assertTrue(insertResponse.ok());
    assertEquals(size, insertResponse.getVectorIds().size());
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

    assertTrue(client.createCollection(collectionMapping).ok());

    List<ByteBuffer> vectors = generateBinaryVectors(size, binaryDimension);
    InsertParam insertParam =
        new InsertParam.Builder(binaryCollectionName).withBinaryVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    assertEquals(size, insertResponse.getVectorIds().size());

    assertTrue(client.dropCollection(binaryCollectionName).ok());
  }

  @Test
  void search() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    vectors = vectors.stream().map(MilvusClientTest::normalizeVector).collect(Collectors.toList());
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    final int searchSize = 5;
    List<List<Float>> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResponse searchResponse = client.search(searchParam);
    assertTrue(searchResponse.ok());
    List<List<Long>> resultIdsList = searchResponse.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResponse.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResponse.QueryResult>> queryResultsList = searchResponse.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());

    final double epsilon = 0.001;
    for (int i = 0; i < searchSize; i++) {
      SearchResponse.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
      assertEquals(vectorIds.get(i), firstQueryResult.getVectorId());
      assertEquals(vectorIds.get(i), resultIdsList.get(i).get(0));
      assertTrue(Math.abs(1 - firstQueryResult.getDistance()) < epsilon);
      assertTrue(Math.abs(1 - resultDistancesList.get(i).get(0)) < epsilon);
    }
  }

  @Test
  void searchAsync() throws ExecutionException, InterruptedException {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    vectors = vectors.stream().map(MilvusClientTest::normalizeVector).collect(Collectors.toList());
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    final int searchSize = 5;
    List<List<Float>> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(randomCollectionName)
            .withFloatVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    ListenableFuture<SearchResponse> searchResponseFuture = client.searchAsync(searchParam);
    SearchResponse searchResponse = searchResponseFuture.get();
    assertTrue(searchResponse.ok());
    List<List<Long>> resultIdsList = searchResponse.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResponse.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResponse.QueryResult>> queryResultsList = searchResponse.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());
    final double epsilon = 0.001;
    for (int i = 0; i < searchSize; i++) {
      SearchResponse.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
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

    assertTrue(client.createCollection(collectionMapping).ok());

    List<ByteBuffer> vectors = generateBinaryVectors(size, binaryDimension);
    InsertParam insertParam =
        new InsertParam.Builder(binaryCollectionName).withBinaryVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(binaryCollectionName).ok());

    final int searchSize = 5;
    List<ByteBuffer> vectorsToSearch = vectors.subList(0, searchSize);

    final long topK = 10;
    SearchParam searchParam =
        new SearchParam.Builder(binaryCollectionName)
            .withBinaryVectors(vectorsToSearch)
            .withTopK(topK)
            .withParamsInJson("{\"nprobe\": 20}")
            .build();
    SearchResponse searchResponse = client.search(searchParam);
    assertTrue(searchResponse.ok());
    List<List<Long>> resultIdsList = searchResponse.getResultIdsList();
    assertEquals(searchSize, resultIdsList.size());
    List<List<Float>> resultDistancesList = searchResponse.getResultDistancesList();
    assertEquals(searchSize, resultDistancesList.size());
    List<List<SearchResponse.QueryResult>> queryResultsList = searchResponse.getQueryResultsList();
    assertEquals(searchSize, queryResultsList.size());
    final double epsilon = 0.001;
    for (int i = 0; i < searchSize; i++) {
      SearchResponse.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
      assertEquals(vectorIds.get(i), firstQueryResult.getVectorId());
      assertEquals(vectorIds.get(i), resultIdsList.get(i).get(0));
    }

    assertTrue(client.dropCollection(binaryCollectionName).ok());
  }

  @Test
  void getCollectionInfo() {
    GetCollectionInfoResponse getCollectionInfoResponse =
        client.getCollectionInfo(randomCollectionName);
    assertTrue(getCollectionInfoResponse.ok());
    assertTrue(getCollectionInfoResponse.getCollectionMapping().isPresent());
    assertEquals(
        getCollectionInfoResponse.getCollectionMapping().get().getCollectionName(),
        randomCollectionName);

    String nonExistingCollectionName = generator.generate(10);
    getCollectionInfoResponse = client.getCollectionInfo(nonExistingCollectionName);
    assertFalse(getCollectionInfoResponse.ok());
    assertFalse(getCollectionInfoResponse.getCollectionMapping().isPresent());
  }

  @Test
  void listCollections() {
    ListCollectionsResponse listCollectionsResponse = client.listCollections();
    assertTrue(listCollectionsResponse.ok());
    assertTrue(listCollectionsResponse.getCollectionNames().contains(randomCollectionName));
  }

  @Test
  void serverStatus() {
    Response serverStatusResponse = client.getServerStatus();
    assertTrue(serverStatusResponse.ok());
  }

  @Test
  void serverVersion() {
    Response serverVersionResponse = client.getServerVersion();
    assertTrue(serverVersionResponse.ok());
  }

  @Test
  void countEntities() {
    insert();
    assertTrue(client.flush(randomCollectionName).ok());

    CountEntitiesResponse countEntitiesResponse = client.countEntities(randomCollectionName);
    assertTrue(countEntitiesResponse.ok());
    assertEquals(size, countEntitiesResponse.getCollectionEntityCount());
  }

  @Test
  void loadCollection() {
    insert();
    assertTrue(client.flush(randomCollectionName).ok());

    Response loadCollectionResponse = client.loadCollection(randomCollectionName);
    assertTrue(loadCollectionResponse.ok());
  }

  @Test
  void getIndexInfo() {
    createIndex();

    GetIndexInfoResponse getIndexInfoResponse = client.getIndexInfo(randomCollectionName);
    assertTrue(getIndexInfoResponse.ok());
    assertTrue(getIndexInfoResponse.getIndex().isPresent());
    assertEquals(getIndexInfoResponse.getIndex().get().getCollectionName(), randomCollectionName);
    assertEquals(getIndexInfoResponse.getIndex().get().getIndexType(), IndexType.IVF_SQ8);
  }

  @Test
  void dropIndex() {
    Response dropIndexResponse = client.dropIndex(randomCollectionName);
    assertTrue(dropIndexResponse.ok());
  }

  @Test
  void getCollectionStats() {
    insert();

    assertTrue(client.flush(randomCollectionName).ok());

    Response getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());

    String jsonString = getCollectionStatsResponse.getMessage();
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
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    GetEntityByIDResponse getEntityByIDResponse =
        client.getEntityByID(randomCollectionName, vectorIds.subList(0, 100));
    assertTrue(getEntityByIDResponse.ok());
    ByteBuffer bb = getEntityByIDResponse.getBinaryVectors().get(0);
    assertTrue(bb == null || bb.remaining() == 0);

    assertArrayEquals(
        getEntityByIDResponse.getFloatVectors().get(0).toArray(), vectors.get(0).toArray());
  }

  @Test
  void getVectorIds() {
    insert();

    assertTrue(client.flush(randomCollectionName).ok());

    Response getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());

    JSONObject jsonInfo = new JSONObject(getCollectionStatsResponse.getMessage());
    JSONObject segmentInfo =
        jsonInfo
            .getJSONArray("partitions")
            .getJSONObject(0)
            .getJSONArray("segments")
            .getJSONObject(0);

    ListIDInSegmentResponse listIDInSegmentResponse =
        client.listIDInSegment(randomCollectionName, segmentInfo.getString("name"));
    assertTrue(listIDInSegmentResponse.ok());
    assertFalse(listIDInSegmentResponse.getIds().isEmpty());
  }

  @Test
  void deleteEntityByID() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    assertTrue(client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, 100)).ok());
    assertTrue(client.flush(randomCollectionName).ok());
    assertEquals(client.countEntities(randomCollectionName).getCollectionEntityCount(), size - 100);
  }

  @Test
  void flush() {
    assertTrue(client.flush(randomCollectionName).ok());
  }

  @Test
  void flushAsync() throws ExecutionException, InterruptedException {
    assertTrue(client.flushAsync(randomCollectionName).get().ok());
  }

  @Test
  void compact() {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    Response getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());

    JSONObject jsonInfo = new JSONObject(getCollectionStatsResponse.getMessage());
    JSONObject segmentInfo =
        jsonInfo
            .getJSONArray("partitions")
            .getJSONObject(0)
            .getJSONArray("segments")
            .getJSONObject(0);

    long previousSegmentSize = segmentInfo.getLong("data_size");

    assertTrue(
        client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, (int) size / 2)).ok());
    assertTrue(client.flush(randomCollectionName).ok());
    assertTrue(client.compact(randomCollectionName).ok());

    getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());
    jsonInfo = new JSONObject(getCollectionStatsResponse.getMessage());
    segmentInfo =
        jsonInfo
            .getJSONArray("partitions")
            .getJSONObject(0)
            .getJSONArray("segments")
            .getJSONObject(0);

    long currentSegmentSize = segmentInfo.getLong("data_size");
    assertTrue(currentSegmentSize < previousSegmentSize);
  }

  @Test
  void compactAsync() throws ExecutionException, InterruptedException {
    List<List<Float>> vectors = generateFloatVectors(size, dimension);
    InsertParam insertParam =
        new InsertParam.Builder(randomCollectionName).withFloatVectors(vectors).build();
    InsertResponse insertResponse = client.insert(insertParam);
    assertTrue(insertResponse.ok());
    List<Long> vectorIds = insertResponse.getVectorIds();
    assertEquals(size, vectorIds.size());

    assertTrue(client.flush(randomCollectionName).ok());

    Response getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());

    JSONObject jsonInfo = new JSONObject(getCollectionStatsResponse.getMessage());
    JSONObject segmentInfo =
        jsonInfo
            .getJSONArray("partitions")
            .getJSONObject(0)
            .getJSONArray("segments")
            .getJSONObject(0);

    long previousSegmentSize = segmentInfo.getLong("data_size");

    assertTrue(
        client.deleteEntityByID(randomCollectionName, vectorIds.subList(0, (int) size / 2)).ok());
    assertTrue(client.flush(randomCollectionName).ok());
    assertTrue(client.compactAsync(randomCollectionName).get().ok());

    getCollectionStatsResponse = client.getCollectionStats(randomCollectionName);
    assertTrue(getCollectionStatsResponse.ok());
    jsonInfo = new JSONObject(getCollectionStatsResponse.getMessage());
    segmentInfo =
        jsonInfo
            .getJSONArray("partitions")
            .getJSONObject(0)
            .getJSONArray("segments")
            .getJSONObject(0);
    long currentSegmentSize = segmentInfo.getLong("data_size");

    assertTrue(currentSegmentSize < previousSegmentSize);
  }
}
