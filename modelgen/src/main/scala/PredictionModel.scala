/**
 * PredictionModel.scala
 * Social media image location prediction based on textual tags.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import scala.util.Random
import java.io.File

import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import breeze.linalg._

class PredictionModel(clusters: KMeansModel, idf: IDFModel,
    classifier: NaiveBayesModel) extends Serializable {

  def predict(tags: Seq[String]): Prediction = {
    val features = idf.transform(tags)
    val (class_, prob) = classifier.predictMultiple(features, 1).head
    val location = clusters.center(class_)
    Prediction((location.lat, location.lon), prob)
  }

  def predictMultiple(tags: Seq[String], count: Int): Seq[Prediction] = {
    val features = idf.transform(tags)
    classifier.predictMultiple(features, count).map({ case (class_, prob) =>
      val location = clusters.center(class_)
      Prediction((location.lat, location.lon), prob)
    })
  }

  def predictHeuristic(tags: Seq[String]): Seq[Prediction] = {
    val features = idf.transform(tags)
    classifier.predictAll(features).
      scanLeft((0, 0.0, 0.0))((b, a) => (a._1, a._2, b._2 + b._3)).
      drop(1).
      takeWhile(_._3 < 0.75).
      map({ case (class_, prob, cumul) =>
        val location = clusters.center(class_)
        Prediction((location.lat, location.lon), prob)
      })
  }

  def validate(m: Media, top: Int): Boolean = {
    val features = idf.transform(m.tags)
    val classes = classifier.predictMultiple(features, top)
    val truth = clusters.predict(m)
    classes.exists(_._1 == truth)
  }

  def validate(media: RDD[Media], top: Int): Double = {
    val bcEngine = media.context.broadcast(this)
    val correct = media.filter(m => bcEngine.value.validate(m, top)).count
    correct.toDouble / media.count.toDouble
  }

}

case class Prediction(center: (Double, Double), probability: Double)

object PredictionModel {

  def crossValidate(samples: RDD[Media], k: Int, top: Int, seed: Long = 42): Double = {

    val folds = samples.mapPartitionsWithIndex { case(index, data) =>
      val rand = new Random(index << 16 ^ seed)
      for (m <- data) yield (rand.nextInt(k), m)
    }

    folds.persist(StorageLevel.MEMORY_AND_DISK)
    val count = folds.count

    var accuracy = 0.0
    for (f <- 0 until k) {
      val training = folds.filter(_._1 != f).map(_._2)
      val testing = folds.filter(_._1 == f).map(_._2)
      val model = train(training, (count * (k - 1) / (k * 1000)).toInt, seed)
      val acc = model.validate(testing, top)
      println(s"Fold $f: Accuracy = $acc")
      accuracy += acc
    }

    folds.unpersist(false)
    println(s"Overall accuracy = ${accuracy / k}")
    accuracy / k
  }

  private def train(samples: RDD[Media], numClasses: Int, seed: Long): PredictionModel = {
    val points = samples.map({ m =>
      val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
      DenseVector(ecef.x, ecef.y, ecef.z) : Vector[Double]
    })
    val clusters = new KMeans(seed = seed).train(points, numClasses)
    val idf = IDF.fit(samples.map(_.tags))
    val training = clusters.predict(samples).
      zip(idf.transform(samples.map(_.tags)))
    val classifier = new NaiveBayes().train(training)

    new PredictionModel(clusters, idf, classifier)
  }

  def train(samples: RDD[Media], numClasses: Option[Int], outputPath: String,
      loadExisting: Boolean = true, seed: Long = 42): PredictionModel = {

    samples.persist(StorageLevel.MEMORY_AND_DISK_SER)

    val actualClasses = numClasses match {
      case Some(n) => n
      case None => (samples.count / 1000).toInt
    }

    val clusters = if (loadExisting && new File(outputPath + ".kmeans").isFile) {
      KMeansModel.load(outputPath + ".kmeans")
    } else {
      val points = samples.map({ m =>
        val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
        DenseVector(ecef.x, ecef.y, ecef.z) : Vector[Double]
      })
      
      val kmeans = new KMeans(seed = seed)
      val clusters = kmeans.train(points, actualClasses)
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

      val nb = new NaiveBayes()
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
