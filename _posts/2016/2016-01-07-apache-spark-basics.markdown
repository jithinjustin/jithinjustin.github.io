---
layout: post
title: "Apache Spark Basics"
date: "2016-01-07 12:34"
---

At a high level, every Spark application consists of a driver program that runs theuser’s main function and executes various parallel operations on a cluster. The main abstraction Spark provides is a **resilient distributed dataset** (RDD), which is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. RDDs are created by starting with a file in the Hadoop file system (or any other Hadoop-supported file system), or an existing Scala collection in the driver program, and transforming it. Users may also ask Spark to persist an RDD in memory, allowing it to be reused efficiently across parallel operations. Finally, RDDs automatically recover from node failures.


A second abstraction in Spark is shared variables that can be used in parallel operations. By default, when Spark runs a function in parallel as a set of tasks on different nodes, it ships a copy of each variable used in the function to each task. Sometimes, a variable needs to be shared across tasks, or between tasks and the driver program. Spark supports two types of shared variables: broadcast variables, which can be used to cache a value in memory on all nodes, and accumulators, which are variables that are only “added” to, such as counters and sums.

All information on this page is taken from official apache spark website for version 1.6.0.

# RDDs Operations

 RDDs support two types of operations: **transformations**, which create a new dataset from an existing one, and **actions**, which return a value to the driver program after running a computation on the dataset. For example, map is a transformation that passes each dataset element through a function and returns a new RDD representing the results. On the other hand, reduce is an action that aggregates all the elements of the RDD using some function and returns the final result to the driver program (although there is also a parallel reduceByKey that returns a distributed dataset).

All transformations in Spark are **lazy**, in that they do not compute their results right away. Instead, they just remember the transformations applied to some base dataset (e.g. a file). The transformations are only computed when an action requires a result to be returned to the driver program. This design enables Spark to run more efficiently – for example, we can realize that a dataset created through map will be used in a reduce and return only the result of the reduce to the driver, rather than the larger mapped dataset.

By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also persist an RDD in memory using the persist (or cache) method, in which case Spark will keep the elements around on the cluster for much faster access the next time you query it. There is also support for persisting RDDs on disk, or replicated across multiple nodes.

# Passing Functions to Spark

Spark’s API relies heavily on passing functions in the driver program to run on the cluster. In Java, functions are represented by classes implementing the interfaces in the <code>org.apache.spark.api.java.function</code> package. There are two ways to create such functions:

* Implement the Function interfaces in your own class, either as an anonymous inner class or a named one, and pass an instance of it to Spark.
* In Java 8, use lambda expressions to concisely define an implementation.


While much of this guide uses lambda syntax for conciseness, it is easy to use all the same APIs in long-form. For example, we could have written our code above as follows:

{% highlight java %}
JavaRDD<String> lines = sc.textFile("data.txt");
JavaRDD<Integer> lineLengths = lines.map(new Function<String, Integer>() {
  public Integer call(String s) { return s.length(); }
});
int totalLength = lineLengths.reduce(new Function2<Integer, Integer, Integer>() {
  public Integer call(Integer a, Integer b) { return a + b; }
});
{% endhighlight %}

Or, if writing the functions inline is unwieldy:

{% highlight java %}
class GetLength implements Function<String, Integer> {
  public Integer call(String s) { return s.length(); }
}
class Sum implements Function2<Integer, Integer, Integer> {
  public Integer call(Integer a, Integer b) { return a + b; }
}

JavaRDD<String> lines = sc.textFile("data.txt");
JavaRDD<Integer> lineLengths = lines.map(new GetLength());
int totalLength = lineLengths.reduce(new Sum());
{% endhighlight %}

Note that anonymous inner classes in Java can also access variables in the enclosing scope as long as they are marked final. Spark will ship copies of these variables to each worker node as it does for other languages.


## Transformations
---

The following table lists some of the common transformations supported by Spark.

<style>

table{
    border-collapse: collapse;
    border-spacing: 0;
    border:2px solid #000000;
}

th{
    border:2px solid #000000;
}

td{
    border:1px solid #000000;
}
</style>

<table class="table">
<tr><th style="width:25%">Transformation</th><th>Meaning</th></tr>
<tr>
  <td> <b>map</b>(<i>func</i>) </td>
  <td> Return a new distributed dataset formed by passing each element of the source through a function <i>func</i>. </td>
</tr>
<tr>
  <td> <b>filter</b>(<i>func</i>) </td>
  <td> Return a new dataset formed by selecting those elements of the source on which <i>func</i> returns true. </td>
</tr>
<tr>
  <td> <b>flatMap</b>(<i>func</i>) </td>
  <td> Similar to map, but each input item can be mapped to 0 or more output items (so <i>func</i> should return a Seq rather than a single item). </td>
</tr>
<tr>
  <td> <b>mapPartitions</b>(<i>func</i>) <a name="MapPartLink"></a> </td>
  <td> Similar to map, but runs separately on each partition (block) of the RDD, so <i>func</i> must be of type
    Iterator&lt;T&gt; =&gt; Iterator&lt;U&gt; when running on an RDD of type T. </td>
</tr>
<tr>
  <td> <b>mapPartitionsWithIndex</b>(<i>func</i>) </td>
  <td> Similar to mapPartitions, but also provides <i>func</i> with an integer value representing the index of
  the partition, so <i>func</i> must be of type (Int, Iterator&lt;T&gt;) =&gt; Iterator&lt;U&gt; when running on an RDD of type T.
  </td>
</tr>
<tr>
  <td> <b>sample</b>(<i>withReplacement</i>, <i>fraction</i>, <i>seed</i>) </td>
  <td> Sample a fraction <i>fraction</i> of the data, with or without replacement, using a given random number generator seed. </td>
</tr>
<tr>
  <td> <b>union</b>(<i>otherDataset</i>) </td>
  <td> Return a new dataset that contains the union of the elements in the source dataset and the argument. </td>
</tr>
<tr>
  <td> <b>intersection</b>(<i>otherDataset</i>) </td>
  <td> Return a new RDD that contains the intersection of elements in the source dataset and the argument. </td>
</tr>
<tr>
  <td> <b>distinct</b>([<i>numTasks</i>])) </td>
  <td> Return a new dataset that contains the distinct elements of the source dataset.</td>
</tr>
<tr>
  <td> <b>groupByKey</b>([<i>numTasks</i>]) <a name="GroupByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, Iterable&lt;V&gt;) pairs. <br />
    <b>Note:</b> If you are grouping in order to perform an aggregation (such as a sum or
      average) over each key, using <code>reduceByKey</code> or <code>aggregateByKey</code> will yield much better
      performance.
    <br />
    <b>Note:</b> By default, the level of parallelism in the output depends on the number of partitions of the parent RDD.
      You can pass an optional <code>numTasks</code> argument to set a different number of tasks.
  </td>
</tr>
<tr>
  <td> <b>reduceByKey</b>(<i>func</i>, [<i>numTasks</i>]) <a name="ReduceByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function <i>func</i>, which must be of type (V,V) =&gt; V. Like in <code>groupByKey</code>, the number of reduce tasks is configurable through an optional second argument. </td>
</tr>
<tr>
  <td> <b>aggregateByKey</b>(<i>zeroValue</i>)(<i>seqOp</i>, <i>combOp</i>, [<i>numTasks</i>]) <a name="AggregateByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, U) pairs where the values for each key are aggregated using the given combine functions and a neutral "zero" value. Allows an aggregated value type that is different than the input value type, while avoiding unnecessary allocations. Like in <code>groupByKey</code>, the number of reduce tasks is configurable through an optional second argument. </td>
</tr>
<tr>
  <td> <b>sortByKey</b>([<i>ascending</i>], [<i>numTasks</i>]) <a name="SortByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs where K implements Ordered, returns a dataset of (K, V) pairs sorted by keys in ascending or descending order, as specified in the boolean <code>ascending</code> argument.</td>
</tr>
<tr>
  <td> <b>join</b>(<i>otherDataset</i>, [<i>numTasks</i>]) <a name="JoinLink"></a> </td>
  <td> When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key.
    Outer joins are supported through <code>leftOuterJoin</code>, <code>rightOuterJoin</code>, and <code>fullOuterJoin</code>.
  </td>
</tr>
<tr>
  <td> <b>cogroup</b>(<i>otherDataset</i>, [<i>numTasks</i>]) <a name="CogroupLink"></a> </td>
  <td> When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (Iterable&lt;V&gt;, Iterable&lt;W&gt;)) tuples. This operation is also called <code>groupWith</code>. </td>
</tr>
<tr>
  <td> <b>cartesian</b>(<i>otherDataset</i>) </td>
  <td> When called on datasets of types T and U, returns a dataset of (T, U) pairs (all pairs of elements). </td>
</tr>
<tr>
  <td> <b>pipe</b>(<i>command</i>, <i>[envVars]</i>) </td>
  <td> Pipe each partition of the RDD through a shell command, e.g. a Perl or bash script. RDD elements are written to the
    process's stdin and lines output to its stdout are returned as an RDD of strings. </td>
</tr>
<tr>
  <td> <b>coalesce</b>(<i>numPartitions</i>) <a name="CoalesceLink"></a> </td>
  <td> Decrease the number of partitions in the RDD to numPartitions. Useful for running operations more efficiently
    after filtering down a large dataset. </td>
</tr>
<tr>
  <td> <b>repartition</b>(<i>numPartitions</i>) </td>
  <td> Reshuffle the data in the RDD randomly to create either more or fewer partitions and balance it across them.
    This always shuffles all data over the network. <a name="RepartitionLink"></a></td>
</tr>
<tr>
  <td> <b>repartitionAndSortWithinPartitions</b>(<i>partitioner</i>) <a name="Repartition2Link"></a></td>
  <td> Repartition the RDD according to the given partitioner and, within each resulting partition,
  sort records by their keys. This is more efficient than calling <code>repartition</code> and then sorting within
  each partition because it can push the sorting down into the shuffle machinery. </td>
</tr>
</table>


## Actions
---
The following table lists some of the common actions supported by Spark.

<table class="table">
<tr><th>Action</th><th>Meaning</th></tr>
<tr>
  <td> <b>reduce</b>(<i>func</i>) </td>
  <td> Aggregate the elements of the dataset using a function <i>func</i> (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel. </td>
</tr>
<tr>
  <td> <b>collect</b>() </td>
  <td> Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data. </td>
</tr>
<tr>
  <td> <b>count</b>() </td>
  <td> Return the number of elements in the dataset. </td>
</tr>
<tr>
  <td> <b>first</b>() </td>
  <td> Return the first element of the dataset (similar to take(1)). </td>
</tr>
<tr>
  <td> <b>take</b>(<i>n</i>) </td>
  <td> Return an array with the first <i>n</i> elements of the dataset. </td>
</tr>
<tr>
  <td> <b>takeSample</b>(<i>withReplacement</i>, <i>num</i>, [<i>seed</i>]) </td>
  <td> Return an array with a random sample of <i>num</i> elements of the dataset, with or without replacement, optionally pre-specifying a random number generator seed.</td>
</tr>
<tr>
  <td> <b>takeOrdered</b>(<i>n</i>, <i>[ordering]</i>) </td>
  <td> Return the first <i>n</i> elements of the RDD using either their natural order or a custom comparator. </td>
</tr>
<tr>
  <td> <b>saveAsTextFile</b>(<i>path</i>) </td>
  <td> Write the elements of the dataset as a text file (or set of text files) in a given directory in the local filesystem, HDFS or any other Hadoop-supported file system. Spark will call toString on each element to convert it to a line of text in the file. </td>
</tr>
<tr>
  <td> <b>saveAsSequenceFile</b>(<i>path</i>) <br /> (Java and Scala) </td>
  <td> Write the elements of the dataset as a Hadoop SequenceFile in a given path in the local filesystem, HDFS or any other Hadoop-supported file system. This is available on RDDs of key-value pairs that implement Hadoop's Writable interface. In Scala, it is also
   available on types that are implicitly convertible to Writable (Spark includes conversions for basic types like Int, Double, String, etc). </td>
</tr>
<tr>
  <td> <b>saveAsObjectFile</b>(<i>path</i>) <br /> (Java and Scala) </td>
  <td> Write the elements of the dataset in a simple format using Java serialization, which can then be loaded using
    <code>SparkContext.objectFile()</code>. </td>
</tr>
<tr>
  <td> <b>countByKey</b>() <a name="CountByLink"></a> </td>
  <td> Only available on RDDs of type (K, V). Returns a hashmap of (K, Int) pairs with the count of each key. </td>
</tr>
<tr>
  <td> <b>foreach</b>(<i>func</i>) </td>
  <td> Run a function <i>func</i> on each element of the dataset. This is usually done for side effects such as updating an <a href="#AccumLink">Accumulator</a> or interacting with external storage systems.
  <br /><b>Note</b>: modifying variables other than Accumulators outside of the <code>foreach()</code> may result in undefined behavior. See <a href="#ClosuresLink">Understanding closures </a> for more details.</td>
</tr>
</table>


## WordCount Example
-----
Wordcount example showing simple transformations and actions for counting number of words in a file.

{% highlight java %}
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples;

import scala.Tuple2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public final class JavaWordCount {
  private static final Pattern SPACE = Pattern.compile(" ");

  public static void main(String[] args) throws Exception {

    if (args.length < 1) {
      System.err.println("Usage: JavaWordCount <file>");
      System.exit(1);
    }

    SparkConf sparkConf = new SparkConf().setAppName("JavaWordCount");
    JavaSparkContext ctx = new JavaSparkContext(sparkConf);
    JavaRDD<String> lines = ctx.textFile(args[0], 1);

    JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
      @Override
      public Iterable<String> call(String s) {
        return Arrays.asList(SPACE.split(s));
      }
    });

    JavaPairRDD<String, Integer> ones = words.mapToPair(new PairFunction<String, String, Integer>() {
      @Override
      public Tuple2<String, Integer> call(String s) {
        return new Tuple2<String, Integer>(s, 1);
      }
    });

    JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Function2<Integer, Integer, Integer>() {
      @Override
      public Integer call(Integer i1, Integer i2) {
        return i1 + i2;
      }
    });

    List<Tuple2<String, Integer>> output = counts.collect();
    for (Tuple2<?,?> tuple : output) {
      System.out.println(tuple._1() + ": " + tuple._2());
    }
    ctx.stop();
  }
}

{% endhighlight %}
