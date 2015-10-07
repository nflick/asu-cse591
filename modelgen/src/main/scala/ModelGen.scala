/**
 * ModelGen.scala
 * Main file for Spark geolocation clustering job.
 * Author: Nathan Flick
 */

import scala.util.parsing.combinator.RegexParsers
import scala.language.postfixOps

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object ModelGen {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: ... ModelGen INPUTFILE OUTPUTFILE NUMCLUSTERS")
      return
    }

    val dataPath = args(0)
    val outPath = args(1)
    val numClusters = args(2).toInt

    val conf = new SparkConf().setAppName("Geolocation Model Generator")
    val sc = new SparkContext(conf)

    val media = sc.textFile(dataPath).
      map(s => CSVLine.parse(s.trim)).
      filter(_.length == 4).
      map(l => Media(l(0).toLong, l(1), l(2).toDouble, l(3).toDouble)).
      cache()

    val clusters = cluster(media, numClusters)
    dumpClusters(media, clusters, outPath)
  }

  def cluster(media: RDD[Media], numClusters: Int) = {
    val ecef = media.map((m: Media) => LLA(m.latitude, m.longitude, 0.0).toECEF)
    val euclidean = ecef.map((e: ECEF) => Vectors.dense(e.x, e.y, e.z)).cache()
    KMeans.train(euclidean, numClusters, 20)
  }

  def dumpClusters(media: RDD[Media], clusters: KMeansModel, path: String) {
    def toClusterId(m: Media) = {
      val ecef = LLA(m.latitude, m.longitude, 0.0).toECEF
      val euclidean = Vectors.dense(ecef.x, ecef.y, ecef.z)
      clusters.predict(euclidean)
    }

    val lines = media.map((m: Media) => s"""${m.id},"${m.tags}",${m.latitude},${m.longitude},${toClusterId(m)}""")
    lines.saveAsTextFile(path)
  }
}

object CSVLine extends RegexParsers {
  def COMMA = ","
  def QUOTE = "\""
  def TXT = "[^\",]".r

  def nonescaped: Parser[String] = (TXT*) ^^ { case ls => ls.mkString("") }
  def escaped: Parser[String] = (QUOTE~>((TXT|COMMA)*)<~QUOTE) ^^ { case ls => ls.mkString("") }
  def field: Parser[String] = escaped | nonescaped

  def record: Parser[List[String]] = (field~((COMMA~>field)*)) ^^ { case f~fs => f::fs }

  def parse(s: String) = parseAll(record, s) match {
    case Success(res, _) => res
    case _ => List[String]()
  }
}