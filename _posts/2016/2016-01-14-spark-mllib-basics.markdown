---
layout: post
title: "Spark MLlib Basics"
date: "2016-01-14 12:10"
---
MLlib is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. It consists of common learning algorithms and utilities, including classification, regression, clustering, collaborative filtering, dimensionality reduction, as well as lower-level optimization primitives and higher-level pipeline APIs.

It divides into two packages:

* <code>spark.mllib </code>contains the original API built on top of RDDs.
* <code>spark.ml</code> provides higher-level API built on top of DataFrames for constructing ML pipelines.


Using <code>spark.ml</code> is recommended because with DataFrames the API is more versatile and flexible.

## MLlib datatypes:

# Local vector

<p>A local vector has integer-typed and 0-based indices and double-typed values, stored on a single
machine.  MLlib supports two types of local vectors: dense and sparse.  A dense vector is backed by
a double array representing its entry values, while a sparse vector is backed by two parallel
arrays: indices and values.  For example, a vector <code>(1.0, 0.0, 3.0)</code> can be represented in dense
format as <code>[1.0, 0.0, 3.0]</code> or in sparse format as <code>(3, [0, 2], [1.0, 3.0])</code>, where <code>3</code> is the size
of the vector.</p>

The base class of local vectors is Vector, and two implementation are provided: DenseVector and SparseVector.
{% highlight java %}
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

// Create a dense vector (1.0, 0.0, 3.0).
Vector dv = Vectors.dense(1.0, 0.0, 3.0);
// Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
{% endhighlight %}


# Labeled point
A labeled point is a local vector, either dense or sparse, associated with a label/response. In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label, so we can use labeled points in both regression and classification. For binary classification, a label should be either 0 (negative) or 1 (positive). For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....

<p>A labeled point is represented by
<code>LabeledPoint</code></p>

{% highlight java %}
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

// Create a labeled point with a positive label and a dense feature vector.
LabeledPoint pos = new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0));

// Create a labeled point with a negative label and a sparse feature vector.
LabeledPoint neg = new LabeledPoint(0.0, Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0}));
{% endhighlight %}

# Main concepts in Pipelines

<p>Spark ML standardizes APIs for machine learning algorithms to make it easier to combine multiple
algorithms into a single pipeline, or workflow.
This section covers the key concepts introduced by the Spark ML API, where the pipeline concept is
mostly inspired by the <a href="http://scikit-learn.org/">scikit-learn</a> project.</p>

**DataFrame**: Spark ML uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types. E.g., a DataFrame could have different columns storing text, feature vectors, true labels, and predictions.

**Transformer**: A Transformer is an algorithm which can transform one DataFrame into another DataFrame. E.g., an ML model is a Transformer which transforms DataFrame with features into a DataFrame with predictions.

**Estimator**: An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

**Pipeline**: A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.

**Parameter**: All Transformers and Estimators now share a common API for specifying parameters.
<p>
Sample code representing Estimators, Transformers and Parameters.
</p>

{% highlight java %}

package com.quixey.sparkpipeline;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;

public class PipelineComponentsTest {

	public static void main(String args[]) {

		SparkConf conf = new SparkConf().setAppName("JavaTokenizerExample")
				.setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(jsc);
		// Prepare training data.
		// We use LabeledPoint, which is a JavaBean. Spark SQL can convert RDDs
		// of JavaBeans
		// into DataFrames, where it uses the bean metadata to infer the schema.
		DataFrame training = sqlContext.createDataFrame(Arrays.asList(
				new LabeledPoint(1.0, Vectors.dense(0.0, 1.1, 0.1)),
				new LabeledPoint(0.0, Vectors.dense(2.0, 1.0, -1.0)),
				new LabeledPoint(0.0, Vectors.dense(2.0, 1.3, 1.0)),
				new LabeledPoint(1.0, Vectors.dense(0.0, 1.2, -0.5))),
				LabeledPoint.class);

		// Create a LogisticRegression instance. This instance is an Estimator.
		LogisticRegression lr = new LogisticRegression();
		// Print out the parameters, documentation, and any default values.
		System.out.println("LogisticRegression parameters:\n"
				+ lr.explainParams() + "\n");

		// We may set parameters using setter methods.
		lr.setMaxIter(10).setRegParam(0.01);

		// Learn a LogisticRegression model. This uses the parameters stored in
		// lr.
		LogisticRegressionModel model1 = lr.fit(training);
		// Since model1 is a Model (i.e., a Transformer produced by an
		// Estimator),
		// we can view the parameters it used during fit().
		// This prints the parameter (name: value) pairs, where names are unique
		// IDs for this
		// LogisticRegression instance.
		System.out.println("Model 1 was fit using parameters: "
				+ model1.parent().extractParamMap());

		// We may alternatively specify parameters using a ParamMap.
		ParamMap paramMap = new ParamMap().put(lr.maxIter().w(20)) // Specify 1
																	// Param.
				.put(lr.maxIter(), 30) // This overwrites the original maxIter.
				.put(lr.regParam().w(0.1), lr.threshold().w(0.55)); // Specify
																	// multiple
																	// Params.

		// One can also combine ParamMaps.
		ParamMap paramMap2 = new ParamMap().put(lr.probabilityCol().w(
				"myProbability")); // Change output column name
		ParamMap paramMapCombined = paramMap.$plus$plus(paramMap2);

		// Now learn a new model using the paramMapCombined parameters.
		// paramMapCombined overrides all parameters set earlier via lr.set*
		// methods.
		LogisticRegressionModel model2 = lr.fit(training, paramMapCombined);
		System.out.println("Model 2 was fit using parameters: "
				+ model2.parent().extractParamMap());

		// Prepare test documents.
		DataFrame test = sqlContext.createDataFrame(Arrays.asList(
				new LabeledPoint(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
				new LabeledPoint(0.0, Vectors.dense(3.0, 2.0, -0.1)),
				new LabeledPoint(1.0, Vectors.dense(0.0, 2.2, -1.5))),
				LabeledPoint.class);

		// Make predictions on test documents using the Transformer.transform()
		// method.
		// LogisticRegression.transform will only use the 'features' column.
		// Note that model2.transform() outputs a 'myProbability' column instead
		// of the usual
		// 'probability' column since we renamed the lr.probabilityCol parameter
		// previously.
		DataFrame results = model2.transform(test);
		for (Row r : results.select("features", "label", "myProbability",
				"prediction").collect()) {
			System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob="
					+ r.get(2) + ", prediction=" + r.get(3));
		}

	}

}
{% endhighlight %}

**Sample output**
<pre>
LogisticRegression parameters:
elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty (default: 0.0)
featuresCol: features column name (default: features)
fitIntercept: whether to fit an intercept term (default: true)
labelCol: label column name (default: label)
maxIter: maximum number of iterations (>= 0) (default: 100)
predictionCol: prediction column name (default: prediction)
probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)
regParam: regularization parameter (>= 0) (default: 0.0)
standardization: whether to standardize the training features before fitting the model (default: true)
threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.5)
thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values >= 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class' threshold. (undefined)
tol: the convergence tolerance for iterative algorithms (default: 1.0E-6)
weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (default: )

Model 1 was fit using parameters: {
	logreg_a16a39642a6b-elasticNetParam: 0.0,
	logreg_a16a39642a6b-featuresCol: features,
	logreg_a16a39642a6b-fitIntercept: true,
	logreg_a16a39642a6b-labelCol: label,
	logreg_a16a39642a6b-maxIter: 10,
	logreg_a16a39642a6b-predictionCol: prediction,
	logreg_a16a39642a6b-probabilityCol: probability,
	logreg_a16a39642a6b-rawPredictionCol: rawPrediction,
	logreg_a16a39642a6b-regParam: 0.01,
	logreg_a16a39642a6b-standardization: true,
	logreg_a16a39642a6b-threshold: 0.5,
	logreg_a16a39642a6b-tol: 1.0E-6,
	logreg_a16a39642a6b-weightCol:
}


Model 2 was fit using parameters: {
	logreg_a16a39642a6b-elasticNetParam: 0.0,
	logreg_a16a39642a6b-featuresCol: features,
	logreg_a16a39642a6b-fitIntercept: true,
	logreg_a16a39642a6b-labelCol: label,
	logreg_a16a39642a6b-maxIter: 30,
	logreg_a16a39642a6b-predictionCol: prediction,
	logreg_a16a39642a6b-probabilityCol: myProbability,
	logreg_a16a39642a6b-rawPredictionCol: rawPrediction,
	logreg_a16a39642a6b-regParam: 0.1,
	logreg_a16a39642a6b-standardization: true,
	logreg_a16a39642a6b-threshold: 0.55,
	logreg_a16a39642a6b-tol: 1.0E-6,
	logreg_a16a39642a6b-weightCol:
}

features         label   predicted vector
([-1.0,1.5,1.3], 1.0) -> prob=[0.05707304171033984,0.9429269582896601], prediction=1.0
([3.0,2.0,-0.1], 0.0) -> prob=[0.9238522311704088,0.0761477688295912], prediction=0.0
([0.0,2.2,-1.5], 1.0) -> prob=[0.10972776114779145,0.8902722388522085], prediction=1.0

</pre>



## Pipeline example

A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage. For Transformer stages, the transform() method is called on the DataFrame. For Estimator stages, the fit() method is called to produce a Transformer (which becomes part of the PipelineModel, or fitted Pipeline), and that Transformer’s transform() method is called on the DataFrame.
{% highlight java %}
import java.io.Serializable;

//Labeled and unlabeled instance types.
//Spark SQL can infer schema from Java Beans.
public class Document implements Serializable {
	private long id;
	private String text;

	public Document(long id, String text) {
		this.id = id;
		this.text = text;
	}

	public long getId() {
		return this.id;
	}

	public void setId(long id) {
		this.id = id;
	}

	public String getText() {
		return this.text;
	}

	public void setText(String text) {
		this.text = text;
	}
}


import java.io.Serializable;

public class LabeledDocument extends Document implements Serializable {
	private double label;

	public LabeledDocument(long id, String text, double label) {
		super(id, text);
		this.label = label;
	}

	public double getLabel() {
		return this.label;
	}

	public void setLabel(double label) {
		this.label = label;
	}
}


import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;

public class PipelineTest {

	public static void printDataFrameasJson(RDD<String> jsonRDD) {

		String[] jarray = (String[]) jsonRDD.collect();
		for (String json : jarray) {
			System.out.println(json);
		}

	}

	public static void main(String args[])

	{
		SparkConf conf = new SparkConf().setAppName("JavaTokenizerExample")
				.setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(jsc);

		// Prepare training documents, which are labeled.
		DataFrame training = sqlContext.createDataFrame(Arrays.asList(
				new LabeledDocument(0L, "a b c d e spark", 1.0),
				new LabeledDocument(1L, "b d", 0.0), new LabeledDocument(2L,
						"spark f g h", 1.0), new LabeledDocument(3L,
						"hadoop mapreduce", 0.0)), LabeledDocument.class);

		// Configure an ML pipeline, which consists of three stages: tokenizer,
		// hashingTF, and lr.
		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol(
				"words");
		HashingTF hashingTF = new HashingTF().setNumFeatures(1000)
				.setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
		LogisticRegression lr = new LogisticRegression().setMaxIter(10)
				.setRegParam(0.01);
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
				tokenizer, hashingTF, lr });

		DataFrame tokenDataFrame = tokenizer.transform(training);
		DataFrame hashingDataFrame = hashingTF.transform(tokenDataFrame);

		printDataFrameasJson(tokenDataFrame.toJSON());
		printDataFrameasJson(hashingDataFrame.toJSON());

		// Fit the pipeline to training documents.
		PipelineModel model = pipeline.fit(training);

		// Prepare test documents, which are unlabeled.
		DataFrame test = sqlContext.createDataFrame(Arrays.asList(new Document(
				4L, "spark i j k"), new Document(5L, "l m n"), new Document(6L,
				"mapreduce spark"), new Document(7L, "apache hadoop")),
				Document.class);

		// Make predictions on test documents.
		DataFrame predictions = model.transform(test);
		for (Row r : predictions.select("id", "text", "probability",
				"prediction").collect()) {
			System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob="
					+ r.get(2) + ", prediction=" + r.get(3));
		}

	}

}
{% endhighlight %}

 **Sample output**

<pre>
Tokenizer
{"id":0,"label":1.0,"text":"a b c d e spark","words":["a","b","c","d","e","spark"]}
{"id":1,"label":0.0,"text":"b d","words":["b","d"]}
{"id":2,"label":1.0,"text":"spark f g h","words":["spark","f","g","h"]}
{"id":3,"label":0.0,"text":"hadoop mapreduce","words":["hadoop","mapreduce"]}

HashingTF
{"id":0,"label":1.0,"text":"a b c d e spark","words":["a","b","c","d","e","spark"],"features":{"type":0,"size":1000,"indices":[97,98,99,100,101,365],"values":[1.0,1.0,1.0,1.0,1.0,1.0]}}
{"id":1,"label":0.0,"text":"b d","words":["b","d"],"features":{"type":0,"size":1000,"indices":[98,100],"values":[1.0,1.0]}}
{"id":2,"label":1.0,"text":"spark f g h","words":["spark","f","g","h"],"features":{"type":0,"size":1000,"indices":[102,103,104,365],"values":[1.0,1.0,1.0,1.0]}}
{"id":3,"label":0.0,"text":"hadoop mapreduce","words":["hadoop","mapreduce"],"features":{"type":0,"size":1000,"indices":[269,810],"values":[1.0,1.0]}}

"id", "text",             "probability",                          "prediction"
(4, spark i j k) --> prob=[0.5406433544851431,0.45935664551485683], prediction=0.0
(5, l m n) --> prob=[0.9334382627383263,0.06656173726167372], prediction=0.0
(6, mapreduce spark) --> prob=[0.7799076868203894,0.2200923131796106], prediction=0.0
(7, apache hadoop) --> prob=[0.9768636139518304,0.023136386048169637], prediction=0.0
</pre>


## Example: model selection via cross-validation
An important task in ML is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Pipelines facilitate model selection by making it easy to tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately.

Currently, spark.ml supports model selection using the CrossValidator class, which takes an Estimator, a set of ParamMaps, and an Evaluator. CrossValidator begins by splitting the dataset into a set of folds which are used as separate training and test datasets; e.g., with k=3k=3 folds, CrossValidator will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. CrossValidator iterates through the set of ParamMaps. For each ParamMap, it trains the given Estimator and evaluates it using the given Evaluator.

The Evaluator can be a RegressionEvaluator for regression problems, a BinaryClassificationEvaluator for binary data, or a MultiClassClassificationEvaluator for multiclass problems. The default metric used to choose the best ParamMap can be overriden by the setMetric method in each of these evaluators.

The ParamMap which produces the best evaluation metric (averaged over the kk folds) is selected as the best model. CrossValidator finally fits the Estimator using the best ParamMap and the entire dataset.

The following example demonstrates using CrossValidator to select from a grid of parameters. To help construct the parameter grid, we use the ParamGridBuilder utility.

Note that cross-validation over a grid of parameters is expensive. E.g., in the example below, the parameter grid has 3 values for hashingTF.numFeatures and 2 values for lr.regParam, and CrossValidator uses 2 folds. This multiplies out to (3×2)×2=12 different models being trained. In realistic settings, it can be common to try many more parameters and use more folds (k=3 and k=10 are common). In other words, using CrossValidator can be very expensive. However, it is also a well-established method for choosing parameters which is more statistically sound than heuristic hand-tuning.
{% highlight java %}
import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

public class PipelinewithCrossvalidation {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		SparkConf conf = new SparkConf().setAppName("JavaTokenizerExample")
				.setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(jsc);
		DataFrame training = sqlContext.createDataFrame(Arrays.asList(
				new LabeledDocument(0L, "a b c d e spark", 1.0),
				new LabeledDocument(1L, "b d", 0.0), new LabeledDocument(2L,
						"spark f g h", 1.0), new LabeledDocument(3L,
						"hadoop mapreduce", 0.0), new LabeledDocument(4L,
						"b spark who", 1.0), new LabeledDocument(5L, "g d a y",
						0.0), new LabeledDocument(6L, "spark fly", 1.0),
				new LabeledDocument(7L, "was mapreduce", 0.0),
				new LabeledDocument(8L, "e spark program", 1.0),
				new LabeledDocument(9L, "a e c l", 0.0), new LabeledDocument(
						10L, "spark compile", 1.0), new LabeledDocument(11L,
						"hadoop software", 0.0)), LabeledDocument.class);

		// Configure an ML pipeline, which consists of three stages: tokenizer,
		// hashingTF, and lr.
		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol(
				"words");
		HashingTF hashingTF = new HashingTF().setNumFeatures(1000)
				.setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
		LogisticRegression lr = new LogisticRegression().setMaxIter(10)
				.setRegParam(0.01);
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
				tokenizer, hashingTF, lr });

		// We use a ParamGridBuilder to construct a grid of parameters to search
		// over.
		// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
		// this grid will have 3 x 2 = 6 parameter settings for CrossValidator
		// to choose from.
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(hashingTF.numFeatures(), new int[] { 10, 100, 1000 })
				.addGrid(lr.regParam(), new double[] { 0.1, 0.01 }).build();

		// We now treat the Pipeline as an Estimator, wrapping it in a
		// CrossValidator instance.
		// This will allow us to jointly choose parameters for all Pipeline
		// stages.
		// A CrossValidator requires an Estimator, a set of Estimator ParamMaps,
		// and an Evaluator.
		// Note that the evaluator here is a BinaryClassificationEvaluator and
		// its default metric
		// is areaUnderROC.
		CrossValidator cv = new CrossValidator().setEstimator(pipeline)
				.setEvaluator(new BinaryClassificationEvaluator())
				.setEstimatorParamMaps(paramGrid).setNumFolds(2); // Use 3+ in
																	// practice

		// Run cross-validation, and choose the best set of parameters.
		CrossValidatorModel cvModel = cv.fit(training);

		// Prepare test documents, which are unlabeled.
		DataFrame test = sqlContext.createDataFrame(Arrays.asList(new Document(
				4L, "spark i j k"), new Document(5L, "l m n"), new Document(6L,
				"mapreduce spark"), new Document(7L, "apache hadoop")),
				Document.class);

		// Make predictions on test documents. cvModel uses the best model found
		// (lrModel).

		DataFrame predictions = cvModel.transform(test);

		for (Row r : predictions.select("id", "text", "probability",
				"prediction").collect()) {
			System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob="
					+ r.get(2) + ", prediction=" + r.get(3));
		}

	}
}
{% endhighlight %}

**Sample Output**

<pre>
(4, spark i j k) --> prob=[0.24804795226775067,0.7519520477322493], prediction=1.0
(5, l m n) --> prob=[0.9647209186740324,0.0352790813259676], prediction=0.0
(6, mapreduce spark) --> prob=[0.4248344997494982,0.5751655002505017], prediction=1.0
(7, apache hadoop) --> prob=[0.6899594200690093,0.3100405799309907], prediction=0.0
</pre>
