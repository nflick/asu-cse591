/**
 * PredictionModel.scala
 * Social media image location prediction based on textual tags.
 * Author: Nathan Flick
 */

package com.github.nflick.models

case class Prediction(center: (Double, Double), probability: Double)

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

}

object PredictionModel {

  def load(path: String): PredictionModel = {
    val clusters = KMeansModel.load(path + ".kmeans")
    val idf = IDFModel.load(path + ".idf")
    val classifier = NaiveBayesModel.load(path + ".nb")
    new PredictionModel(clusters, idf, classifier)
  }

}