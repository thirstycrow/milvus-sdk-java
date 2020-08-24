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
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import io.grpc.StatusRuntimeException;
import io.milvus.client.exception.MilvusServerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

/** Actual implementation of interface <code>MilvusClient</code> */
public class MilvusGrpcClient implements MilvusClient {
  private static final Logger logger = LoggerFactory.getLogger(MilvusGrpcClient.class);

  private final JavaFlavorMilvusClient client;

  ////////////////////// Constructor //////////////////////
  MilvusGrpcClient(JavaFlavorMilvusClient client) {
    this.client = client;
  }

  @Override
  public void close(long maxWaitSeconds) {
    client.close();
  }

  private boolean maybeAvailable() {
    switch (client.getGrpcState()) {
      case IDLE:
      case CONNECTING:
      case READY:
        return true;
      default:
        return false;
    }
  }

  public MilvusClient withTimeout(long timeout, TimeUnit unit) {
    return new MilvusGrpcClient(client.withTimeout(timeout, unit));
  }

  @Override
  public Response createCollection(@Nonnull CollectionMapping collectionMapping) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Response.CLIENT_NOT_CONNECTED;
    }

    try {
      client.createCollection(collectionMapping);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      if (e.getReason().contentEquals("Collection already exists")) {
        logWarning("Collection `{}` already exists", collectionMapping.getCollectionName());
      } else {
        logError("Create collection failed\n{}\n{}", collectionMapping.toString(), e.toString());
      }
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("createCollection RPC failed:\n{}", e.getStatus().toString());
      return new Response(e);
    }
  }

  @Override
  public HasCollectionResponse hasCollection(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new HasCollectionResponse(Response.CLIENT_NOT_CONNECTED, false);
    }

    try {
      boolean result = client.hasCollection(collectionName);
      return new HasCollectionResponse(Response.SUCCESS, result);
    } catch (MilvusServerException e) {
      logError("hasCollection `{}` failed:\n{}", collectionName, e.getReason());
      return new HasCollectionResponse(new Response(e), false);
    } catch (StatusRuntimeException e) {
      logError("hasCollection RPC failed:\n{}", e.getStatus().toString());
      return new HasCollectionResponse(new Response(e), false);
    }
  }

  @Override
  public Response dropCollection(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Response.CLIENT_NOT_CONNECTED;
    }

    try {
      client.dropCollection(collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Drop collection `{}` failed:\n{}", collectionName, e.getErrorCode());
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("dropCollection RPC failed:\n{}", e.getStatus().toString());
      return new Response(e);
    }
  }

  @Override
  public Response createIndex(@Nonnull Index index) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Response.CLIENT_NOT_CONNECTED;
    }

    try {
      client.createIndex(index);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Create index failed:\n{}\n{}", index.toString(), e.getReason());
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("createIndex RPC failed:\n{}", e.getStatus().toString());
      return new Response(e);
    }
  }

  @Override
  public ListenableFuture<Response> createIndexAsync(@Nonnull Index index) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Futures.immediateFuture(Response.CLIENT_NOT_CONNECTED);
    }

    ListenableFuture<Void> response = client.createIndexAsync(index);

    Futures.addCallback(
        response,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void v) {
            logInfo("Created index successfully!\n{}", index);
          }

          @Override
          public void onFailure(Throwable t) {
            if (t instanceof MilvusServerException) {
              logError("CreateIndexAsync failed:\n{}\n{}", index, t);
            } else {
              logError("CreateIndexAsync failed:\n{}", t.getMessage());
            }
          }
        },
        MoreExecutors.directExecutor());

    return Futures.transform(response, v -> Response.SUCCESS, MoreExecutors.directExecutor());
  }

  @Override
  public Response createPartition(String collectionName, String tag) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.createPartition(collectionName, tag);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Create partition `{}` in collection `{}` failed: {}", tag, collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("createPartition RPC failed:\n{}", e.getStatus().toString());
      return new Response(e);
    }
  }

  @Override
  public HasPartitionResponse hasPartition(String collectionName, String tag) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new HasPartitionResponse(new Response(Response.Status.CLIENT_NOT_CONNECTED), false);
    }

    try {
      boolean result = client.hasPartition(collectionName, tag);
      logInfo("hasPartition with tag `{}` in `{}` = {}", tag, collectionName, result);
      return new HasPartitionResponse(Response.SUCCESS, result);
    } catch (MilvusServerException e) {
      logError("hasPartition with tag `{}` in `{}` failed:\n{}", tag, collectionName, e);
      return new HasPartitionResponse(new Response(e), false);
    } catch (StatusRuntimeException e) {
      logError("hasPartition RPC failed:\n{}", e.getStatus().toString());
      return new HasPartitionResponse(new Response(e), false);
    }
  }

  @Override
  public ListPartitionsResponse listPartitions(String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new ListPartitionsResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList());
    }

    try {
      List<String> partitions = client.listPartitions(collectionName);
      logInfo("Current partitions of collection {}: {}", collectionName, partitions);
      return new ListPartitionsResponse(Response.SUCCESS, partitions);
    } catch (MilvusServerException e) {
      logError("List partitions failed:\n{}", e);
      return new ListPartitionsResponse(new Response(e), Collections.emptyList());
    } catch (StatusRuntimeException e) {
      logError("listPartitions RPC failed:\n{}", e.getStatus());
      return new ListPartitionsResponse(new Response(e), Collections.emptyList());
    }
  }

  @Override
  public Response dropPartition(String collectionName, String tag) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.dropPartition(collectionName, tag);
      logInfo("Dropped partition `{}` in collection `{}` successfully!", tag, collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Drop partition `{}` in collection `{}` failed:\n{}", tag, collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("dropPartition RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public InsertResponse insert(@Nonnull InsertParam insertParam) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new InsertResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList());
    }

    try {
      List<Long> ids = client.insert(insertParam);
      logInfo("Inserted {} vectors to collection `{}` successfully!",
          ids.size(), insertParam.getCollectionName());
      return new InsertResponse(Response.SUCCESS, ids);
    } catch (MilvusServerException e) {
      logError("Insert vectors failed:\n{}", e);
      return new InsertResponse(new Response(e), Collections.emptyList());
    } catch (StatusRuntimeException e) {
      logError("insert RPC failed:\n{}", e.getStatus());
      return new InsertResponse(new Response(e), Collections.emptyList());
    }
  }

  @Override
  public ListenableFuture<InsertResponse> insertAsync(@Nonnull InsertParam insertParam) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Futures.immediateFuture(
          new InsertResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList()));
    }

    ListenableFuture<List<Long>> response = client.insertAsync(insertParam);

    Futures.addCallback(
        response,
        new FutureCallback<List<Long>>() {
          @Override
          public void onSuccess(List<Long> result) {
            logInfo("Inserted {} vectors to collection `{}` successfully!",
                result.size(), insertParam.getCollectionName());
          }

          @Override
          public void onFailure(Throwable t) {
            logError("InsertAsync failed:\n{}", t);
          }
        },
        MoreExecutors.directExecutor());

    return Futures.transform(response,
        ids -> new InsertResponse(Response.SUCCESS, ids),
        MoreExecutors.directExecutor());
  }

  @Override
  public SearchResponse search(@Nonnull SearchParam searchParam) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      SearchResponse searchResponse = new SearchResponse();
      searchResponse.setResponse(Response.CLIENT_NOT_CONNECTED);
      return searchResponse;
    }

    try {
      SearchResult result = client.search(searchParam);
      SearchResponse searchResponse = buildSearchResponse(result);
      searchResponse.setResponse(Response.SUCCESS);
      logInfo(
          "Search completed successfully! Returned results for {} queries",
          searchResponse.getNumQueries());
      return searchResponse;
    } catch (MilvusServerException e) {
      logError("Search failed:\n{}", e);
      SearchResponse searchResponse = new SearchResponse();
      searchResponse.setResponse(new Response(e));
      return searchResponse;
    } catch (StatusRuntimeException e) {
      logError("search RPC failed:\n{}", e.getStatus().toString());
      SearchResponse searchResponse = new SearchResponse();
      searchResponse.setResponse(new Response(e));
      return searchResponse;
    }
  }

  @Override
  public ListenableFuture<SearchResponse> searchAsync(@Nonnull SearchParam searchParam) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      SearchResponse searchResponse = new SearchResponse();
      searchResponse.setResponse(Response.CLIENT_NOT_CONNECTED);
      return Futures.immediateFuture(searchResponse);
    }

    ListenableFuture<SearchResult> response = client.searchAsync(searchParam);

    Futures.addCallback(
        response,
        new FutureCallback<SearchResult>() {
          @Override
          public void onSuccess(SearchResult result) {
            logInfo(
                  "SearchAsync completed successfully! Returned results for {} queries",
                  result.getNumQueries());
          }

          @Override
          public void onFailure(Throwable t) {
            logError("SearchAsync failed:\n{}", t);
          }
        },
        MoreExecutors.directExecutor());

    return Futures.transform(response, this::buildSearchResponse, MoreExecutors.directExecutor());
  }

  @Override
  public GetCollectionInfoResponse getCollectionInfo(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new GetCollectionInfoResponse(
          new Response(Response.Status.CLIENT_NOT_CONNECTED), null);
    }

    try {
      CollectionMapping collectionMapping = client.getCollectionInfo(collectionName);
      logInfo("Get Collection Info `{}` returned:\n{}", collectionName, collectionMapping);
      return new GetCollectionInfoResponse(Response.SUCCESS, collectionMapping);
    } catch (MilvusServerException e) {
      logError("Get Collection Info `{}` failed:\n{}", collectionName, e);
      return new GetCollectionInfoResponse(new Response(e), null);
    } catch (StatusRuntimeException e) {
      logError("getCollectionInfo RPC failed:\n{}", e.getStatus());
      return new GetCollectionInfoResponse(new Response(e), null);
    }
  }

  @Override
  public ListCollectionsResponse listCollections() {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new ListCollectionsResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList());
    }

    try {
      List<String> collectionNames = client.listCollections();
      logInfo("Current collections: {}", collectionNames.toString());
      return new ListCollectionsResponse(Response.SUCCESS, collectionNames);
    } catch (MilvusServerException e) {
      logError("List collections failed:\n{}", e);
      return new ListCollectionsResponse(new Response(e), Collections.emptyList());
    } catch (StatusRuntimeException e) {
      logError("listCollections RPC failed:\n{}", e.getStatus());
      return new ListCollectionsResponse(new Response(e), Collections.emptyList());
    }
  }

  @Override
  public CountEntitiesResponse countEntities(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new CountEntitiesResponse(new Response(Response.Status.CLIENT_NOT_CONNECTED), 0);
    }

    try {
      long collectionRowCount = client.countEntities(collectionName);
      logInfo("Collection `{}` has {} entities", collectionName, collectionRowCount);
      return new CountEntitiesResponse(Response.SUCCESS, collectionRowCount);
    } catch (MilvusServerException e) {
      logError("Get collection `{}` entity count failed:\n{}", collectionName, e);
      return new CountEntitiesResponse(new Response(e), 0);
    } catch (StatusRuntimeException e) {
      logError("countEntities RPC failed:\n{}", e.getStatus().toString());
      return new CountEntitiesResponse(new Response(e), 0);
    }
  }

  @Override
  public Response getServerStatus() {
    return command("status");
  }

  @Override
  public Response getServerVersion() {
    return command("version");
  }

  public Response command(@Nonnull String command) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Response.CLIENT_NOT_CONNECTED;
    }

    try {
      String reply = client.command(command);
      logInfo("Command `{}`: {}", command, reply);
      return new Response(Response.Status.SUCCESS, reply);
    } catch (MilvusServerException e) {
      logError("Command `{}` failed:\n{}", command, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("Command RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public Response loadCollection(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.loadCollection(collectionName);
      logInfo("Loaded collection `{}` successfully!", collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Load collection `{}` failed:\n{}", collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("loadCollection RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public GetIndexInfoResponse getIndexInfo(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new GetIndexInfoResponse(new Response(Response.Status.CLIENT_NOT_CONNECTED), null);
    }

    try {
      Index index = client.getIndexInfo(collectionName);
      logInfo("Get index info for collection `{}` returned:\n{}", collectionName, index);
      return new GetIndexInfoResponse(Response.SUCCESS, index);
    } catch (MilvusServerException e) {
      logError("Get index info for collection `{}` failed:\n{}", collectionName, e);
      return new GetIndexInfoResponse(new Response(e), null);
    } catch (StatusRuntimeException e) {
      logError("getIndexInfo RPC failed:\n{}", e.getStatus());
      return new GetIndexInfoResponse(new Response(e), null);
    }
  }

  @Override
  public Response dropIndex(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.dropIndex(collectionName);
      logInfo("Dropped index for collection `{}` successfully!", collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Drop index for collection `{}` failed:\n{}", collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("dropIndex RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public Response getCollectionStats(String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      String result = client.getCollectionStats(collectionName);
      logInfo("getCollectionStats for `{}` returned successfully!", collectionName);
      return new Response(Response.Status.SUCCESS, result);
    } catch (MilvusServerException e) {
      logError("getCollectionStats for `{}` failed:\n{}", collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("getCollectionStats RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public GetEntityByIDResponse getEntityByID(String collectionName, List<Long> ids) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new GetEntityByIDResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList(), null);
    }

    try {
      VectorEntities response = client.getEntityByID(collectionName, ids);
      logInfo("getEntityByID in collection `{}` returned successfully!", collectionName);
      return new GetEntityByIDResponse(Response.SUCCESS, response.getFloatVectors(), response.getBinaryVectors());
    } catch (MilvusServerException e) {
      logError("getEntityByID in collection `{}` failed:\n{}", collectionName, e);
      return new GetEntityByIDResponse(new Response(e), Collections.emptyList(), null);
    } catch (StatusRuntimeException e) {
      logError("getEntityByID RPC failed:\n{}", e.getStatus());
      return new GetEntityByIDResponse(new Response(e), Collections.emptyList(), null);
    }
  }

  @Override
  public ListIDInSegmentResponse listIDInSegment(String collectionName, String segmentName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new ListIDInSegmentResponse(Response.CLIENT_NOT_CONNECTED, Collections.emptyList());
    }

    try {
      List<Long> ids = client.listIDInSegment(collectionName, segmentName);
      logInfo("listIDInSegment in collection `{}`, segment `{}` returned successfully!",
          collectionName, segmentName);
      return new ListIDInSegmentResponse(Response.SUCCESS, ids);
    } catch (MilvusServerException e) {
      logError("listIDInSegment in collection `{}`, segment `{}` failed:\n{}",
          collectionName, segmentName, e);
      return new ListIDInSegmentResponse(new Response(e), Collections.emptyList());
    } catch (StatusRuntimeException e) {
      logError("listIDInSegment RPC failed:\n{}", e.getStatus());
      return new ListIDInSegmentResponse(new Response(e), Collections.emptyList());
    }
  }

  @Override
  public Response deleteEntityByID(String collectionName, List<Long> ids) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.deleteEntityByID(collectionName, ids);
      logInfo("deleteEntityByID in collection `{}` completed successfully!", collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("deleteEntityByID in collection `{}` failed:\n{}", collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("deleteEntityByID RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public Response flush(List<String> collectionNames) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.flush(collectionNames);
      logInfo("Flushed collection {} successfully!", collectionNames);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Flush collection {} failed:\n{}", collectionNames, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("flush RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public ListenableFuture<Response> flushAsync(@Nonnull List<String> collectionNames) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Futures.immediateFuture(new Response(Response.Status.CLIENT_NOT_CONNECTED));
    }

    ListenableFuture<Void> response = client.flushAsync(collectionNames);

    Futures.addCallback(
        response,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void result) {
            logInfo("Flushed collection {} successfully!", collectionNames);
          }

          @Override
          public void onFailure(Throwable t) {
            if (t instanceof MilvusServerException) {
              logError("Flush collection {} failed:\n{}", collectionNames, t);
            } else {
              logError("FlushAsync failed:\n{}", t.getMessage());
            }
          }
        },
        MoreExecutors.directExecutor());

    return Futures.transform(response, v -> Response.SUCCESS, MoreExecutors.directExecutor());
  }

  @Override
  public Response flush(String collectionName) {
    return flush(ImmutableList.of(collectionName));
  }

  @Override
  public ListenableFuture<Response> flushAsync(String collectionName) {
    return flushAsync(ImmutableList.of(collectionName));
  }

  @Override
  public Response compact(String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return new Response(Response.Status.CLIENT_NOT_CONNECTED);
    }

    try {
      client.compact(collectionName);
      logInfo("Compacted collection `{}` successfully!", collectionName);
      return Response.SUCCESS;
    } catch (MilvusServerException e) {
      logError("Compact collection `{}` failed:\n{}", collectionName, e);
      return new Response(e);
    } catch (StatusRuntimeException e) {
      logError("compact RPC failed:\n{}", e.getStatus());
      return new Response(e);
    }
  }

  @Override
  public ListenableFuture<Response> compactAsync(@Nonnull String collectionName) {
    if (!maybeAvailable()) {
      logWarning("You are not connected to Milvus server");
      return Futures.immediateFuture(Response.CLIENT_NOT_CONNECTED);
    }

    ListenableFuture<Void> response = client.compactAsync(collectionName);

    Futures.addCallback(
        response,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void result) {
            logInfo("Compacted collection `{}` successfully!", collectionName);
          }

          @Override
          public void onFailure(Throwable t) {
            if (t instanceof MilvusServerException) {
              logError("Compact collection `{}` failed:\n{}", collectionName, t);
            } else {
              logError("CompactAsync failed:\n{}", t.getMessage());
            }
          }
        },
        MoreExecutors.directExecutor());

    return Futures.transform(response, v -> Response.SUCCESS, MoreExecutors.directExecutor());
  }

  ///////////////////// Util Functions/////////////////////
  private SearchResponse buildSearchResponse(SearchResult searchResult) {
    SearchResponse searchResponse = new SearchResponse();
    searchResponse.setResponse(Response.SUCCESS);
    searchResponse.setNumQueries(searchResult.getNumQueries());
    searchResponse.setTopK(searchResult.getTopK());
    searchResponse.setResultIdsList(searchResult.getResultIdsList());
    searchResponse.setResultDistancesList(searchResult.getResultDistancesList());
    return searchResponse;
  }

  ///////////////////// Log Functions//////////////////////

  protected void logInfo(String msg, Object... params) {
    logger.info(msg, params);
  }

  protected void logWarning(String msg, Object... params) {
    logger.warn(msg, params);
  }

  protected void logError(String msg, Object... params) {
    logger.error(msg, params);
  }
}
