/**
 * KDTree.scala
 * KDTree for use in optimized KMeans algorithm.
 * Author: Nathan Flick
 */

import org.apache.spark.mllib.linalg.{Vector, Vectors}

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