/**
 * KMeans.scala
 * Optimized Naive Bayes using Sparse matrices.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import breeze.linalg._

class NaiveBayes(lambda: Double) {

  def train(data: RDD[(Int, SparseVector[Double])]) = {
    // Adapted from https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/classification/NaiveBayes.scala.
    // Optimized for large numbers of classes and features through the use of sparse vectors
    // to store the conditional probabilities of features for each class and a reordering
    // of the mathematical operations to increase storage efficiency by maximizing the number of
    // zero entries.

    val aggregated = data.combineByKey[(Long, SparseVector[Double])](
      createCombiner = (v: SparseVector[Double]) => (1L, v),
      mergeValue = (c: (Long, SparseVector[Double]), v: SparseVector[Double]) => (c._1 + 1L, c._2 :+= v),
      mergeCombiners = (c1: (Long, SparseVector[Double]), c2: (Long, SparseVector[Double])) => (c1._1 + c2._1, c1._2 :+= c2._2)
    ).collect().sortBy(_._1)

    val numLabels = aggregated.length
    var numDocuments = 0L
    aggregated foreach { case (_, (n, _)) =>
      numDocuments += n
    }

    val numFeatures = aggregated.head match { case (_, (_, v)) => v.size }

    val labels = new Array[Int](numLabels)
    val pi = new Array[Double](numLabels)
    val thetaLogDenom = new Array[Double](numLabels)
    
    val piLogDenom = math.log(numDocuments + numLabels * lambda)
    val theta = Array.tabulate[SparseVector[Double]](numLabels)({ i => 
      aggregated(i) match { case (label, (n, sumTermFreqs)) =>
        labels(i) = label
        pi(i) = math.log(n + lambda) - piLogDenom
        thetaLogDenom(i) = math.log(sum(sumTermFreqs) + numFeatures * lambda)

        for (k <- sumTermFreqs.activeKeysIterator) {
          // Subtracing thetaLogDenom will take place later to maximize zero entries and optimize
          // memory usage of the sparse vectors.
          sumTermFreqs(k) = math.log(sumTermFreqs(k) + lambda)
        }

        sumTermFreqs
      }
    })

    new NaiveBayesModel(labels, pi, theta, thetaLogDenom)
  }

}

class NaiveBayesModel(val labels: Array[Int], val pi: Array[Double],
    val theta: Array[SparseVector[Double]], val thetaLogDenom: Array[Double])
    extends Serializable {

  def predict(features: SparseVector[Double]): Int = {
    val logProb = multinomial(features)
    val index = logProb.indexOf(logProb.max)
    labels(index)
  }

  def predict(samples: RDD[SparseVector[Double]]): RDD[Int] = {
    val bcModel = samples.context.broadcast(this)
    samples map { v =>
      bcModel.value.predict(v)
    }
  }

  private def multinomial(features: SparseVector[Double]): Array[Double] = {
    Array.tabulate[Double](theta.length)(i =>
      product(features, theta(i), thetaLogDenom(i)) + pi(i))
  }

  private def product(features: SparseVector[Double], probs: SparseVector[Double],
      denom: Double) = {

    require(features.size == probs.size, "Vectors are different sizes")
    var product = 0.0
    var offset = 0
    while (offset < features.activeSize) {
      product += features.valueAt(offset) * (probs(features.indexAt(offset)) - denom)
      offset += 1
    }

    product
  }

  private def posteriorProbabilities(logProb: Array[Double]) = {
    val maxLog = logProb.max
    val scaledProbs = logProb.map(lp => math.exp(lp - maxLog))
    val probSum = scaledProbs.sum
    scaledProbs.map(_ / probSum)
  }

  def save(path: String) = {
    val objStream = new ObjectOutputStream(new FileOutputStream(path))
    try {
      objStream.writeObject(this)
    } finally {
      objStream.close()
    }
  }
}

object NaiveBayesModel {

  def load(path: String) = {
    val objStream = new ObjectInputStream(new FileInputStream(path))
    try {
      val model = objStream.readObject() match {
        case m: NaiveBayesModel => m
        case other => throw new ClassCastException(s"Expected NaiveBayesModel, got ${other.getClass}.")
      }
      model
    } finally {
      objStream.close()
    }
  }

}