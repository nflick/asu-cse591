/**
 * ModelGen.scala
 * Main file for project.
 * Author: Nathan Flick
 */

package com.github.nflick.modelgen

import scala.util.parsing.combinator.RegexParsers
import scala.language.postfixOps
import scala.util.Random

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

case class Arguments(
  appName: String = "Geolocation Model Builder",
  command: Symbol = null,
  numClasses: Option[Int] = None,
  loadExisting: Boolean = true,
  seed: Long = 42,
  sourcePath: String = null,
  modelPath: String = null,
  outputPath: String = null,
  tags: String = null
)

object ModelGen {
  
  val argParser = new scopt.OptionParser[Arguments]("modelgen") {
    head("ModelGen", "0.0.1")
    
    opt[String]('a', "appname").
      valueName("<name>").
      action((x, c) => c.copy(appName = x)).
      text("Specify the Spark App Name.")

    note("\n")
    cmd("visualize").
      text("Generate KML file to visualize the location classes.").
      action((_, c) => c.copy(command = 'visualize)).
      children(
        arg[String]("<model>").
          action((x, c) => c.copy(modelPath = x)).
          text("The partition model."),

        arg[String]("<output>").
          action((x, c) => c.copy(outputPath = x)).
          text("Location to save the KML file.")
      )

    note("\n")
    cmd("train").
      text("Train the prediction model.").
      action((_, c) => c.copy(command = 'train)).
      children(
        opt[Int]('n', "num-classes").
          valueName("<num>").
          action((x, c) => c.copy(numClasses = Some(x))).
          text("Number of classes (default 1 per 1000 samples)."),

        opt[Unit]('i', "ignore-existing").
          action((_, c) => c.copy(loadExisting = false)).
          text("Do not used already generated components of model."),

        opt[Int]('s', "seed").
          valueName("<value>").
          action((x, c) => c.copy(seed = x)).
          text("Seed for random number generator."),

        arg[String]("<source>").
          action((x, c) => c.copy(sourcePath = x)).
          text("Source of samples in CSV format."),

        arg[String]("<output>").
          action((x, c) => c.copy(modelPath = x)).
          text("Location to save the generated model.")
      )

    note("\n")
    cmd("predict").
      text("Predict the location of a set of tags.").
      action((_, c) => c.copy(command = 'predict)).
      children(
        arg[String]("<model>").
          action((x, c) => c.copy(modelPath = x)).
          text("The tag prediction model."),

        arg[String]("<tags>").
          action((x, c) => c.copy(tags = x)).
          text("The set of tags to predict as {tag1,tag2...}.")
      )

    checkConfig(c => if (c.command == null) failure("A command must be provided.") else success)
  }

  def main(args: Array[String]): Unit = {
    
    argParser.parse(args, Arguments()) match {
      case Some(arguments) => {
        val conf = new SparkConf().setAppName(arguments.appName)
        val sc = new SparkContext(conf)

        arguments.command match {
          case 'visualize => visualize(sc, arguments)
          case 'train => train(sc, arguments)
          case 'predict => predict(sc, arguments)
          case _ =>
        }
      }

      case None =>
    }
  }

  def visualize(sc: SparkContext, args: Arguments) = {
    val rand = new Random()
    def randomColor() = {
      val bytes = Array.ofDim[Byte](3)
      rand.nextBytes(bytes)
      f"ff${bytes(0)}%02x${bytes(1)}%02x${bytes(2)}%02x"
    }

    val model = KMeansModel.load(args.modelPath + ".kmeans")

    val kml =
    <kml><Document>
      { for (i <- 0 until model.numCenters) yield
        <Style id={i.toString}><IconStyle>
        <color>{randomColor()}</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle></Style>
      } { for (i <- 0 until model.numCenters) yield
        <Placemark>
        <styleUrl>{i.toString}</styleUrl>
        <Point><coordinates>{s"${model.center(i).lat},${model.center(i).lon}"}</coordinates></Point>
        </Placemark>
    } </Document></kml>
    
    scala.xml.XML.save(args.outputPath, kml)
  }

  def train(sc: SparkContext, args: Arguments): Unit = {
    val samples = loadSamples(sc, args.sourcePath).cache()
    val numClasses = args.numClasses match {
      case Some(n) => n
      case None => samples.count / 1000
    }

    val engine = PredictionModel.train(samples, numClasses.toInt, args.modelPath,
      args.loadExisting, args.seed)
  }

  def predict(sc: SparkContext, args: Arguments) = {
    val tags = TagList.parse(args.tags)
    val engine = PredictionModel.load(args.modelPath)
    val location = engine.predict(tags)
    println(s"PREDICTION: ${location.toString}")
  }

  def loadSamples(sc: SparkContext, path: String): RDD[Media] = {
    sc.textFile(path).
      repartition(20).
      mapPartitions({ rows =>
        val format = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
        for {
          row <- rows.map(r => CSVLine.parse(r.trim))
          if row.length == 8
        } yield {
          Media(row(0).toLong, row(1).toLong, LocalDateTime.parse(row(2), format), TagList.parse(row(3)),
            row(4).toLong, row(5), row(6).toDouble, row(7).toDouble)
        }
      })
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
    case Success(res, _) => res
    case _ => List[String]()
  }
}
