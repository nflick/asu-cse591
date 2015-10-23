/**
 * LocationPartitioner.scala
 * Partitioners for grouping coordinates into discrete cells.
 * Author: Nathan Flick
 */

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

trait LocationPartitioner {
  def partition(media: Media): Int
  def partition(media: RDD[Media]): RDD[Int]
  def partitionCenter(partition: Int): LLA
  def save(sc: SparkContext, path: String)
}

class KMeansPartitioner(model: KMeansModel)
  extends LocationPartitioner {

  def partition(media: Media): Int = {
    val ecef = LLA(media.latitude, media.longitude, 0.0).toECEF
    val euclidean = Vectors.dense(ecef.x, ecef.y, ecef.z)
    model.predict(euclidean)
  }

  def partition(media: RDD[Media]): RDD[Int] = {
    def vec(m: Media) = {
      val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
      Vectors.dense(ecef.x, ecef.y, ecef.z)
    }

    model.predict(media.map(m => {
      val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
      Vectors.dense(ecef.x, ecef.y, ecef.z)
    }))
  }

  def partitionCenter(partition: Int) = {
    val vec = model.clusterCenters(partition)
    ECEF(vec(0), vec(1), vec(2)).toLLA
  }

  def save(sc: SparkContext, path: String) = {
    model.save(sc, path)
  }

}

object KMeansPartitioner {

  def build(media: RDD[Media], numClusters: Int) = {
    val ecef = media.map(m => LLA(m.latitude, m.longitude, 0.0).toECEF)
    val euclidean = ecef.map(e => Vectors.dense(e.x, e.y, e.z)).cache()
    val model = KMeans.train(euclidean, numClusters, 20)
    new KMeansPartitioner(model)
  }

  def load(sc: SparkContext, path: String) = {
    new KMeansPartitioner(KMeansModel.load(sc, path))
  }

}

