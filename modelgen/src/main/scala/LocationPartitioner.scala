/**
 * LocationPartitioner.scala
 * Partitioners for grouping coordinates into discrete cells.
 * Author: Nathan Flick
 */

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Random

trait LocationPartitioner {
  def partition(media: Media): Int
  def partition(media: RDD[Media]): RDD[Int]
  def numPartitions: Int
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
    maxIterations: Int = 20, seed: Long = 42) = {
    
    val ecef = media.map(m => LLA(m.latitude, m.longitude, 0.0).toECEF)
    val data = ecef.map(e => Vectors.dense(e.x, e.y, e.z)).cache()
    val centers = runAlgorithm(data, numClusters, maxIterations, seed)
    new KMeansPartitioner(new KMeansModel(centers))
  }
    
  private def addVecs(a: Vector, b: Vector) = {
    val c = Array.ofDim[Double](a.size)
    for ( i <- 0 until a.size ) c(i) = a(i) + b(i)
    Vectors.dense(c)
  }

  private def scalVec(a: Vector, b: Double) = {
    val c = Array.ofDim[Double](a.size)
    for ( i <- 0 until a.size ) c(i) = a(i) * b
    Vectors.dense(c)
  }

  private def runAlgorithm(data: RDD[Vector], numClusters: Int,
    maxIterations: Int = 20, seed: Long = 42, epsilon: Double = 1e-5): Array[Vector] = {

    // Custom implementation of KMeans using KDTree optimizations.
    val rand = new Random(seed)
    var centers = data.takeSample(true, numClusters, rand.nextLong())
    var iteration = 0
    var changed = true

    while (iteration < maxIterations && changed) {
      type WeightedPoint = (Vector, Long)
      def mergeContribs(x: WeightedPoint, y: WeightedPoint) = {
        (addVecs(x._1, y._1), x._2 + y._2)
      }

      val kdtree = KDTree((0 until centers.size).map(i => (centers(i), i))) match {
        case Some(t) => t
        case None => throw new IllegalArgumentException("Cannot build KDTree with no centers.")
      }

      val bcTree = data.context.broadcast(kdtree)
      val bcCenters = data.context.broadcast(centers)
      val costAccum = data.context.accumulator(0.0)

      val totalContribs = data.mapPartitions { points =>
        val centers = bcCenters.value
        val tree = bcTree.value
        val dims = centers(0).size
        val sums = Array.fill(centers.length)(Vectors.zeros(dims))
        val counts = Array.fill(centers.length)(0L)

        points.foreach { point =>
          val (bestCenter, bestIndex) = tree.nearest(point) match {
            case KDTree.Nearest(key, value, to) => (key, value)
          }

          costAccum += Vectors.sqdist(point, bestCenter)
          sums(bestIndex) = addVecs(sums(bestIndex), bestCenter)
          counts(bestIndex) += 1
        }

        val contribs = for ( k <- 0 until centers.length ) yield {
          (k, (sums(k), counts(k)))
        }
        contribs.iterator

      }.reduceByKey { (x, y) =>
        (addVecs(x._1, y._1), x._2 + y._2)
      }.collectAsMap()

      changed = false
      for ( k <- 0 until centers.length ) {
        val (sum, count) = totalContribs(k)
        if (count != 0) {
          val newCenter = scalVec(sum, 1.0 / count.toDouble)
          if (Vectors.sqdist(newCenter, centers(k)) > epsilon * epsilon) { changed = true }
          centers(k) = newCenter
        }
      }

      iteration += 1
    }

    centers
  }

  def load(sc: SparkContext, path: String) = {
    new KMeansPartitioner(KMeansModel.load(sc, path))
  }

}

