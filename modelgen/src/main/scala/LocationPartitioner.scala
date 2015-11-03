/**
 * LocationPartitioner.scala
 * Partitioners for grouping coordinates into discrete cells.
 * Author: Nathan Flick
 */


import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

trait LocationPartitioner {
  def partition(media: Media): Int
  def partition(media: RDD[Media]): RDD[Int]
  def numPartitions: Int
  def partitionCenter(partition: Int): LLA
  def save(sc: SparkContext, path: String)
}

class KMeansPartitioner(model: KMeansModel)
  extends LocationPartitioner {

  private val kdtree = KDTree(model.clusterCenters.zipWithIndex) match {
    case Some(tree) => tree
    case None => throw new IllegalArgumentException("Model must have at least one center.")
  }

  def partition(media: Media): Int = {
    val ecef = LLA(media.latitude, media.longitude, 0.0).toECEF
    val euclidean = Vectors.dense(ecef.x, ecef.y, ecef.z)
    kdtree.nearest(euclidean).tag
  }

  def partition(media: RDD[Media]): RDD[Int] = {
    val bcTree = media.context.broadcast(kdtree)
    media map { m =>
      val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
      val vec = Vectors.dense(ecef.x, ecef.y, ecef.z)
      bcTree.value.nearest(vec).tag
    }
  }

  def numPartitions: Int = model.clusterCenters.length

  def partitionCenter(partition: Int) = {
    val vec = model.clusterCenters(partition)
    ECEF(vec(0), vec(1), vec(2)).toLLA
  }

  def save(sc: SparkContext, path: String) = {
    model.save(sc, path)
  }

}

object KMeansPartitioner {

  def build(media: RDD[Media], numClusters: Int, 
    maxIterations: Int = 40, seed: Long = 42) = {
    val ecef = media.map(m => LLA(m.latitude, m.longitude, 0.0).toECEF)
    val data = ecef.map(e => Vectors.dense(e.x, e.y, e.z)).cache()
    val kmeans = new KMeans(data, numClusters, maxIterations = maxIterations, seed = seed)
    val centers = kmeans.runAlgorithm()
    new KMeansPartitioner(new KMeansModel(centers))
  }

  def load(sc: SparkContext, path: String) = {
    new KMeansPartitioner(KMeansModel.load(sc, path))
  }

}

