/**
 * ModelGen.scala
 * Main file for Spark geolocation clustering job.
 * Author: Nathan Flick
 */

import scala.util.parsing.combinator.RegexParsers
import scala.language.postfixOps

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

case class Arguments(
  appName: String = "Geolocation Model Builder",
  command: Symbol = null,
  numPartitions: Option[Int] = None,
  sourcePath: String = null,
  partitionPath: String = null,
  latitude: Double = 0.0,
  longitude: Double = 0.0
)

object ModelGen {
  
  val argParser = new scopt.OptionParser[Arguments]("modelgen") {
    head("ModelGen", "0.0.1")
    
    opt[String]('a', "appname").
      valueName("<name>").
      action((x, c) => c.copy(appName = x)).
      text("Specify the Spark App Name.")

    note("\n")
    cmd("partition").
      text("Generate the location partitions/cells.").
      action((_, c) => c.copy(command = 'partition)).
      children (
        opt[Int]('n', "num-partitions").
          valueName("<num>").
          action((x, c) => c.copy(numPartitions = Some(x))).
          text("Number of partitions (default 1 per 1000 samples)."),

        arg[String]("<source>").
          action((x, c) => c.copy(sourcePath = x)).
          text("Source of samples in CSV format."),

        arg[String]("<output>").
          action((x, c) => c.copy(partitionPath = x)).
          text("Location to save generated partitions.")
      )

    note("\n")
    cmd("predict").
      text("Predict the partition for the given location.").
      action((_, c) => c.copy(command = 'predict)).
      children(
        arg[String]("<partitions>").
          action((x, c) => c.copy(partitionPath = x)).
          text("The partition model."),

        arg[Double]("<lat>").
          action((x, c) => c.copy(latitude = x)).
          text("Latitude for prediction."),

        arg[Double]("<long>").
          action((x, c) => c.copy(longitude = x)).
          text("Longitude for prediction.")
      )

    checkConfig(c => if (c.command == null) failure("A command must be provided.") else success)
  }

  def main(args: Array[String]): Unit = {
    
    argParser.parse(args, Arguments()) match {
      case Some(arguments) => {
        val conf = new SparkConf().setAppName(arguments.appName)
        val sc = new SparkContext(conf)

        arguments.command match {
          case 'partition => partition(sc, arguments)
          case 'predict => predict(sc, arguments)
          case _ =>
        }
      }

      case None =>
    }
  }

  def partition(sc: SparkContext, args: Arguments) = {
    val samples = loadSamples(sc, args.sourcePath)
    val partitions = args.numPartitions match {
      case Some(p) => p
      case None => samples.count() / 1000
    }

    val model = KMeansPartitioner.build(samples, partitions.toInt)
    model.save(sc, args.partitionPath)
  }

  def predict(sc: SparkContext, args: Arguments) = {
    val model = KMeansPartitioner.load(sc, args.partitionPath)
    val partition = model.partition(
      Media(0, 0, LocalDateTime.now(), Array[String](), 0, "", args.latitude, args.longitude))
    println(partition)
  }

  def loadSamples(sc: SparkContext, path: String) = {
    sc.textFile(path).
      repartition(20).
      map(s => CSVLine.parse(s.trim)).
      filter(_.length == 8).
      map(l => Media(l(0).toLong, l(1).toLong, 
        LocalDateTime.parse(l(2), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
        TagList.parse(l(3)),
        l(4).toLong, l(5), l(6).toDouble, l(7).toDouble)).
      filter(_.tags.length > 0)
  }
}

object CSVLine extends RegexParsers {
  override def skipWhitespace = false
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

object TagList extends RegexParsers {
  def LBRACE = "{"
  def RBRACE = "}"
  def COMMA = ","
  def TXT = "[^{},]".r

  def tag: Parser[String] = (TXT*) ^^ { case ls => ls.mkString("") }
  def record: Parser[List[String]] = (LBRACE~>(tag~((COMMA~>tag)*))<~RBRACE) ^^ { case f~fs => f::fs }

  def parse(s: String) = parseAll(record, s) match {
    case Success(res, _) => res.toArray
    case _ => Array[String]()
  }
}

