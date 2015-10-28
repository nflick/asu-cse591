/**
 * KDTreeSpec.scala
 * Unit tests for KDTree.
 * Author: Nathan Flick
 */

import org.scalatest._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Random

class KDTreeSpec extends FlatSpec with Matchers {

  "KDTree" should "return None for an empty sequence" in {
    val tree = KDTree(List())
    tree should equal (None)
  }

  it should "build a large tree" in {
    val rand = new Random(42)  
    val points = for ( i <- 1 to 10000 ) yield {
      Vectors.dense(rand.nextDouble(), rand.nextDouble(), rand.nextDouble)
    }

    val tree = KDTree[Int](points.map((_, 0))).getOrElse(null)

    def bruteNearest(to: Vector, in: Seq[Vector]) = {
      in.minBy(Vectors.sqdist(_, to))
    }

    (1 to 1000).
      map(i => Vectors.dense(rand.nextDouble(), rand.nextDouble(), rand.nextDouble())).
      foreach({ pt =>
        tree.nearest(pt) should equal (KDTree.Nearest(bruteNearest(pt, points), 0, pt))
      })
  }

  it should "perform well" in {
    val rand = new Random(42)  
    val points = for ( i <- 1 to 7000 ) yield {
      Vectors.dense(rand.nextDouble(), rand.nextDouble(), rand.nextDouble)
    }

    val tree = KDTree[Int](points.map((_, 0))).getOrElse(null)

    def bruteNearest(to: Vector, in: Seq[Vector]) = {
      in.minBy(Vectors.sqdist(_, to))
    }

    (1 to 300000).
      map(i => Vectors.dense(rand.nextDouble(), rand.nextDouble(), rand.nextDouble())).
      foreach({ pt =>
        tree.nearest(pt)
        //bruteNearest(pt, points)
      })
  }

}
