package io.milvus.client;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SearchResult {
  private int numQueries;
  private long topK;
  private List<List<Long>> resultIdsList;
  private List<List<Float>> resultDistancesList;

  public SearchResult(int numQueries,
                      long topK,
                      List<List<Long>> resultIdsList,
                      List<List<Float>> resultDistancesList) {
    this.numQueries = numQueries;
    this.topK = topK;
    this.resultIdsList = resultIdsList;
    this.resultDistancesList = resultDistancesList;
  }

  public int getNumQueries() {
    return numQueries;
  }

  public long getTopK() {
    return topK;
  }

  public List<List<Long>> getResultIdsList() {
    return resultIdsList;
  }

  public List<List<Float>> getResultDistancesList() {
    return resultDistancesList;
  }

  /**
   * @return a <code>List</code> of <code>QueryResult</code>s. Each inner <code>List</code> contains
   *     the query result of a vector.
   */
  public List<List<QueryResult>> getQueryResultsList() {
    return IntStream.range(0, numQueries).mapToObj(i ->
        IntStream.range(0, resultIdsList.get(i).size()).mapToObj(j ->
            new QueryResult(
                resultIdsList.get(i).get(j),
                resultDistancesList.get(i).get(j)))
            .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  public static class QueryResult {
    private final long vectorId;
    private final float distance;

    QueryResult(long vectorId, float distance) {
      this.vectorId = vectorId;
      this.distance = distance;
    }

    public long getVectorId() {
      return vectorId;
    }

    public float getDistance() {
      return distance;
    }
  }
}
