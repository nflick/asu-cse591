/**
 * IDF.scala
 * Inverse document frequency calculations.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import java.io._

import org.apache.spark.rdd._
import breeze.linalg._

object IDF {

  def fit(docs: RDD[Seq[String]], minOccurences: Int = 5): IDFModel = {
    val total = docs.count()
    val terms = docs.flatMap(identity).
      countByValue().
      filter(_._2 > minOccurences).
      toSeq.zipWithIndex.map({ case ((tag, count), index) =>
        tag -> (index -> math.log(total.toDouble / count.toDouble))
      }).toMap

    new IDFModel(terms)
  }

}

@SerialVersionUID(1L)
class IDFModel(terms: Map[String, (Int, Double)]) extends Serializable {

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

  def save(path: String): Unit = {
    val objStream = new ObjectOutputStream(new FileOutputStream(path))
    try {
      objStream.writeObject(this)
    } finally {
      objStream.close()
    }
  }

}

object IDFModel {

  def load(path: String): IDFModel = {
    val objStream = new ObjectInputStream(new FileInputStream(path))
    try {
      val model = objStream.readObject() match {
        case m: IDFModel => m
        case other => throw new ClassCastException(s"Expected IDFModel, got ${other.getClass}.")
      }
      model
    } finally {
      objStream.close()
    }
  }

}
