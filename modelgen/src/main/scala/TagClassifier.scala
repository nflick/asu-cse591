/**
 * TagClassifer.scala
 * Naive Bayes classification of tags to locations.
 * Author: Nathan Flick
 */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import breeze.linalg._

import java.io._

class TagClassifier(val model: NaiveBayesModel, val tfidfModel: TFIDFModel) {

  def predict(tags: List[String]) = {
    val features = tfidfModel.transform(tags)
    model.predict(features)
  }

  def save(sc: SparkContext, path: String) = {
    model.save(path)
    tfidfModel.save(path + ".idf")
  }

}

class TFIDFModel(val terms: Map[String, (Int, Double)]) extends Serializable {

  def transform(features: Seq[String]): SparseVector[Double] = {
    val builder = new VectorBuilder[Double](terms.size)
    features.foreach({ f =>
      terms.get(f) match {
        case Some((index, tfidf)) => builder.add(index, tfidf)
        case None =>
      }
    })
    builder.toSparseVector()
  }

  def transform(features: RDD[Seq[String]]): RDD[SparseVector[Double]] = {
    val bcModel = features.context.broadcast(this)
    features map { seq =>
      val model = bcModel.value
      model.transform(seq)
    }
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

object TFIDF {

  def fit(docs: RDD[Seq[String]], minOccurences: Int = 5) = {
    val total = docs.count()
    val terms = docs.flatMap(identity).
      countByValue().
      filter(_._2 > minOccurences).
      toSeq.zipWithIndex.map({ case ((tag, count), index) =>
        tag -> (index -> math.log(total.toDouble / count.toDouble))
      }).toMap

      new TFIDFModel(terms)
  }

  def load(path: String) = {
    val objStream = new ObjectInputStream(new FileInputStream(path))
    try {
      val model = objStream.readObject() match {
        case m: TFIDFModel => m
        case other => throw new ClassCastException(s"Expected TFIDFModel, got ${other.getClass}.")
      }
      model
    } finally {
      objStream.close()
    }
  }

}

object TagClassifier {

  def train(media: RDD[Media], partitioner: LocationPartitioner) = {
    val tfidfModel = TFIDF.fit(media.map(_.tags))
    val training = partitioner.partition(media).
      zip(tfidfModel.transform(media.map(_.tags)))

    val nb = new NaiveBayes(lambda = 1.0)
    val model = nb.train(training)
    new TagClassifier(model, tfidfModel)
  }

  def load(sc: SparkContext, path: String) = {
    val model = NaiveBayes.load(path)
    val tfidfModel = TFIDF.load(path + ".idf")
    new TagClassifier(model, tfidfModel)
  }

}
