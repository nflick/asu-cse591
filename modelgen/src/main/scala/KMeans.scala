/**
 * KMeans.scala
 * Optimized KMeans using KDTrees.
 * Author: Nathan Flick
 */

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel

object KDTree {

  def apply[T](points: Seq[(Vector, T)], depth: Int = 0): Option[KDNode[T]] = {
    val dim = points.headOption match {
      case None => 0
      case Some((v, t)) => v.size
    }

    if (points.isEmpty || dim < 1) None
    else {
      val axis = depth % dim
      val sorted = points.sortBy(_._1(axis))
      val median = sorted(sorted.size / 2)._1(axis)
      val (left, right) = sorted.partition(_._1(axis) < median)
      Some(KDNode(right.head._1, right.head._2, apply(left, depth + 1), apply(right.tail, depth + 1), axis))
    }
  }

  case class KDNode[T](key: Vector, value: T,
    left: Option[KDNode[T]], right: Option[KDNode[T]], axis: Int)
    extends Serializable {

    def nearest(to: Vector): Nearest[T] = {
      val default = Nearest[T](key, value, to)
      (0 until to.size).map(i => to(i) - key(i)).find(_ != 0.0).getOrElse(0.0) match {
        case 0.0 => default
        case t =>
          lazy val bestL = left.map(_.nearest(to)).getOrElse(default)
          lazy val bestR = right.map(_.nearest(to)).getOrElse(default)
          val branch1 = if (t < 0) bestL else bestR
          val best = if (branch1.sqdist < default.sqdist) branch1 else default
          val splitDist = to(axis) - key(axis)

          if (splitDist * splitDist < best.sqdist) {
            val branch2 = if (t < 0) bestR else bestL
            if (branch2.sqdist < best.sqdist) branch2 else best
          } else best
      }
    }

  }

  case class Nearest[T](key: Vector, value: T, to: Vector) {
    lazy val sqdist = Vectors.sqdist(key, to)
  }

}

private object VectorImplicits {

  implicit class ExtendedVector(val v: Vector) {
    
    def +(u: Vector): Vector = {
      require(v.size == u.size, "Vectors are different sizes.")
      val y = Array.ofDim[Double](v.size)
      for ( i <- 0 until v.size ) y(i) = v(i) + u(i)
      Vectors.dense(y)
    }

    def *(a: Double): Vector = {
      val y = Array.ofDim[Double](v.size)
      for (i <- 0 until v.size ) y(i) = v(i) * a
      Vectors.dense(y)
    }

  }

}

class KMeans(data: RDD[Vector], numClusters: Int,
  maxIterations: Int = 25, initSteps: Int = 5,
  seed: Long = 42, epsilon: Double = 1e-5) {

  import VectorImplicits._

  def runAlgorithm(): Array[Vector] = {

    // Custom implementation of KMeans using KDTree optimizations.
    val rand = new Random(seed)
    var centers = initParallel()
    var iteration = 0
    var changed = true

    while (iteration < maxIterations && changed) {
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
          val (center, index) = tree.nearest(point) match {
            case KDTree.Nearest(key, value, to) => (key, value)
          }

          costAccum += Vectors.sqdist(point, center)
          sums(index) = sums(index) + center
          counts(index) += 1
        }

        val contribs = for ( k <- 0 until centers.length ) yield {
          (k, (sums(k), counts(k)))
        }
        contribs.iterator

      }.reduceByKey { (x, y) =>
        (x._1 + y._1, x._2 + y._2)
      }.collectAsMap()

      changed = false
      for ( k <- 0 until centers.length ) {
        val (sum, count) = totalContribs(k)
        if (count != 0) {
          val newCenter = sum * (1.0 / count.toDouble)
          if (Vectors.sqdist(newCenter, centers(k)) > epsilon * epsilon) { changed = true }
          centers(k) = newCenter
        }
      }

      iteration += 1
    }

    centers
  }

  def initParallel(): Array[Vector] = {
    val centers = ArrayBuffer.empty[Vector]
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize the first center to a random point.
    val rand = new Random(seed)
    val sample = data.takeSample(true, 1, rand.nextLong()).toSeq.head
    val newCenters = ArrayBuffer(sample)

    var step = 0
    while (step < initSteps) {
      val kdtree = KDTree((0 until newCenters.size).map(i => (newCenters(i), i))) match {
        case Some(t) => t
        case None => throw new IllegalArgumentException("Cannot build KDTree with no centers.")
      }

      val bcTree = data.context.broadcast(kdtree)
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        val nearest = bcTree.value.nearest(point)
        math.min(Vectors.sqdist(nearest.key, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)

      val sumCosts = costs.aggregate(0.0)((u, v) => u + v, (u, v) => u + v)
      preCosts.unpersist(blocking = false)

      val seed = rand.nextInt()
      val k = numClusters
      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val r = new Random(seed ^ (step << 16) ^ index)
        pointsWithCosts.flatMap { case (p, c) =>
          if (r.nextDouble() < 2.0 * c * k / sumCosts) Some(p) else None
        }
      }.collect()

      centers ++= newCenters
      newCenters.clear()

      chosen.foreach { case (p) =>
        newCenters += p.toDense
      }

      step += 1
    }

    centers ++= newCenters
    newCenters.clear()
    costs.unpersist(blocking = false)

    val kdtree = KDTree((0 until centers.size).map(i => (centers(i), i))) match {
      case Some(t) => t
      case None => throw new IllegalArgumentException("Cannot build KDTree with no centers.")
    }

    val bcTree = data.context.broadcast(kdtree)
    val bcCenters = data.context.broadcast(centers)

    val finalCenters = data.map { p =>
      val nearest = bcTree.value.nearest(p)
      (nearest.value, 1)
    }.reduceByKey(_ + _).
      sortBy(_._2, ascending = false).
      map(t => bcCenters.value(t._1)).
      take(numClusters)

    assert(finalCenters.length == numClusters,
      s"Got ${finalCenters.length} centers from InitParallel, expected ${numClusters}.")
    finalCenters
  }
}