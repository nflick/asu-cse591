/**
 * PredictionModel.scala
 * Social media image location prediction based on textual tags.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import java.io.File

import org.apache.spark.rdd._
import breeze.linalg._

class PredictionModel(clusters: KMeansModel, idf: IDFModel,
    classifier: NaiveBayesModel) extends Serializable {

  def predict(tags: Seq[String]): LLA = {
    val features = idf.transform(tags)
    val class_ = classifier.predict(features)
    clusters.center(class_)
  }

  def validate(m: Media): Boolean = {
    val features = idf.transform(m.tags)
    val class_ = classifier.predict(features)
    class_ == clusters.predict(m)
  }

  def validate(media: RDD[Media]): Double = {
    val bcEngine = media.context.broadcast(this)
    val correct = media.filter(m => bcEngine.value.validate(m)).count
    correct.toDouble / media.count.toDouble
  }

}

object PredictionModel {

  def train(samples: RDD[Media], numClasses: Int, outputPath: String, loadExisting: Boolean = true,
      seed: Long = 42): PredictionModel = {

    samples.persist()

    val clusters = if (loadExisting && new File(outputPath + ".kmeans").isFile) {
      KMeansModel.load(outputPath + ".kmeans")
    } else {
      val points = samples.map({ m =>
        val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
        DenseVector(ecef.x, ecef.y, ecef.z) : Vector[Double]
      })
      
      val kmeans = new KMeans(seed = seed)
      val clusters = kmeans.train(points, numClasses)
      clusters.save(outputPath + ".kmeans")
      clusters
    }

    val idf = if (loadExisting && new File(outputPath + ".idf").isFile) {
      IDFModel.load(outputPath + ".idf")
    } else {
      val idf = IDF.fit(samples.map(_.tags))
      idf.save(outputPath + ".idf")
      idf
    }

    val classifier = if (loadExisting && new File(outputPath + ".nb").isFile) {
      NaiveBayesModel.load(outputPath + ".nb")
    } else {
      val training = clusters.predict(samples).
        zip(idf.transform(samples.map(_.tags)))

      val nb = new NaiveBayes(lambda = 1.0)
      val classifier = nb.train(training)
      classifier.save(outputPath + ".nb")
      classifier
    }

    samples.unpersist(false)
    new PredictionModel(clusters, idf, classifier)
  }

  def load(path: String): PredictionModel = {
    val clusters = KMeansModel.load(path + ".kmeans")
    val idf = IDFModel.load(path + ".idf")
    val classifier = NaiveBayesModel.load(path + ".nb")
    new PredictionModel(clusters, idf, classifier)
  }

}
