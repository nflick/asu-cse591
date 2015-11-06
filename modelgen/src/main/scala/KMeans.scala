/**
 * KMeans.scala
 * Optimized KMeans using KDTrees.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import breeze.linalg._

trait VectorDist {

  def squareDist(v1: Vector[Double], v2: Vector[Double]): Double = {
    val s = v1.size
    require(s == v2.size, "Vector sizes must be equal")

    var i = 0
    var dist = 0.0
    while (i < s) {
      val score = v1(i) - v2(i)
      dist += score * score
      i += 1
    }

    dist
  }

}

object KDTree {

  def apply[T](points: Seq[(Vector[Double], T)], depth: Int = 0): Option[KDNode[T]] = {
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

  case class KDNode[T](value: Vector[Double], tag: T,
      left: Option[KDNode[T]], right: Option[KDNode[T]], axis: Int) extends Serializable {

    def nearest(to: Vector[Double]): Nearest[T] = {
      val default = Nearest[T](value, tag, to)
      val dist = to(axis) - value(axis)

      lazy val bestL = left.map(_.nearest(to)).getOrElse(default)
      lazy val bestR = right.map(_.nearest(to)).getOrElse(default)
      val branch1 = if (dist < 0) bestL else bestR
      val best = if (branch1.sqdist < default.sqdist) branch1 else default

      if (dist * dist < best.sqdist) {
        val branch2 = if (dist < 0) bestR else bestL
        if (branch2.sqdist < best.sqdist) branch2 else best
      } else best
    }

  }

  case class Nearest[T](value: Vector[Double], tag: T, to: Vector[Double]) extends VectorDist {
    val sqdist = squareDist(value, to)
  }

}

class KMeans(maxIterations: Int = 50, initSteps: Int = 5,
  seed: Long = 42, epsilon: Double = 1e-5) extends VectorDist {

  def train(data: RDD[Vector[Double]], numClusters: Int): KMeansModel = {
    // Adapted from https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala
    // Optimized for spatial data through the use of KD-Trees for faster lookups
    // of the nearest center.

    data.persist()
    val rand = new Random(seed)
    var centers = initParallel(data, numClusters)
    var iteration = 0
    var changed = true

    while (iteration < maxIterations && changed) {
      val kdtree = KDTree(centers.zipWithIndex) match {
        case Some(t) => t
        case None => throw new IllegalArgumentException("Cannot build KDTree with zero centers.")
      }

      val bcTree = data.context.broadcast(kdtree)
      val bcCenters = data.context.broadcast(centers)
      val costAccum = data.context.accumulator(0.0)

      val totalContribs = data.mapPartitions { points =>
        val centers = bcCenters.value
        val tree = bcTree.value
        val dims = centers(0).size
        val sums = Array.fill(centers.length)(DenseVector.zeros[Double](dims))
        val counts = Array.fill(centers.length)(0L)

        points.foreach { point =>
          val nearest = tree.nearest(point)
          costAccum += nearest.sqdist
          sums(nearest.tag) += nearest.value
          counts(nearest.tag) += 1
        }

        val contribs = for ( k <- 0 until centers.length ) yield {
          (k, (sums(k), counts(k)))
        }
        contribs.iterator

      }.reduceByKey { (x, y) =>
        (x._1 :+= y._1, x._2 + y._2)
      }.collectAsMap()

      changed = false
      for (k <- 0 until centers.length) {
        val (sum, count) = totalContribs(k)
        if (count != 0) {
          val newCenter = sum * (1.0 / count.toDouble)
          if (squareDist(newCenter, centers(k)) > epsilon * epsilon) { changed = true }
          centers(k) = newCenter
        }
      }

      iteration += 1
    }

    data.unpersist(false)
    new KMeansModel(centers)
  }

  private def initParallel(data: RDD[Vector[Double]], numClusters: Int):
      Array[Vector[Double]] = {
    // Adapted from https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala.

    val centers = ArrayBuffer.empty[Vector[Double]]
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize the first center to a random point.
    val rand = new Random(seed)
    val sample = data.takeSample(true, 1, rand.nextLong()).toSeq.head.toDenseVector
    val newCenters = ArrayBuffer[Vector[Double]](sample)

    var step = 0
    while (step < initSteps) {
      val kdtree = KDTree((0 until newCenters.size).map(i => (newCenters(i), i))) match {
        case Some(t) => t
        case None => throw new IllegalStateException("Cannot build KDTree with zero centers.")
      }

      val bcTree = data.context.broadcast(kdtree)
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        val nearest = bcTree.value.nearest(point)
        math.min(nearest.squareDist(nearest.value, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)

      val sumCosts = costs.aggregate(0.0)((u, v) => u + v, (u, v) => u + v)
      preCosts.unpersist(blocking = false)

      val seed = rand.nextInt()
      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val r = new Random(seed ^ (step << 16) ^ index)
        pointsWithCosts.flatMap { case (p, c) =>
          if (r.nextDouble() < 2.0 * c * numClusters / sumCosts) Some(p) else None
        }
      }.collect()

      centers ++= newCenters
      newCenters.clear()

      chosen.foreach { case (p) =>
        newCenters += p.toDenseVector
      }

      step += 1
    }

    centers ++= newCenters
    newCenters.clear()
    costs.unpersist(blocking = false)

    val kdtree = KDTree(centers.zipWithIndex) match {
      case Some(t) => t
      case None => throw new IllegalArgumentException("Cannot build KDTree with zero centers.")
    }

    val bcTree = data.context.broadcast(kdtree)
    val bcCenters = data.context.broadcast(centers)

    val finalCenters = data.map { p =>
      val nearest = bcTree.value.nearest(p)
      (nearest.tag, 1)
    }.reduceByKey(_ + _).
      sortBy(_._2, ascending = false).
      map(t => bcCenters.value(t._1)).
      take(numClusters)

    assert(finalCenters.length == numClusters,
      s"Got ${finalCenters.length} centers from InitParallel, expected ${numClusters}.")
    finalCenters
  }
}

class KMeansModel(centers: Array[Vector[Double]]) extends Serializable {

  private val kdtree = KDTree(centers.zipWithIndex) match {
    case Some(t) => t
    case None => throw new IllegalArgumentException("Cannot construct KDTree with zero points")
  }

  def predict(m: Media): Int = {
    val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
    val v = DenseVector(ecef.x, ecef.y, ecef.z)
    predict(v)
  }

  def predict(v: Vector[Double]): Int = kdtree.nearest(v).tag

  def predict(media: RDD[Media]): RDD[Int] = {
    val bcModel = media.context.broadcast(this)
    media map { m =>
      bcModel.value.predict(m)
    }
  }

  def center(label: Int): LLA = {
    val v = centers(label)
    ECEF(v(0), v(1), v(2)).toLLA
  }

  def numCenters: Int = centers.length

  def save(path: String): Unit = {
    val objStream = new ObjectOutputStream(new FileOutputStream(path))
    try {
      objStream.writeObject(this)
    } finally {
      objStream.close()
    }
  }

}

object KMeansModel {

  def load(path: String): KMeansModel = {
    val objStream = new ObjectInputStream(new FileInputStream(path))
    try {
      objStream.readObject() match {
        case m: KMeansModel => m
        case other => throw new ClassCastException(s"Expected KMeansModel, got ${other.getClass}.")
      }
    } finally {
      objStream.close()
    }
  }

}
