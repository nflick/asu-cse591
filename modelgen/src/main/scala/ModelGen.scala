/**
 * ModelGen.scala
 * Main file for Spark geolocation clustering job.
 * Author: Nathan Flick
 */

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
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

case class Arguments(
  appName: String = "Geolocation Model Builder",
  command: Symbol = null,
  numPartitions: Option[Int] = None,
  sourcePath: String = null,
  partitionPath: String = null,
  outputPath: String = null,
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
    cmd("visualize").
      text("Generate KML file to visualize the location partitions/cells.").
      action((_, c) => c.copy(command = 'visualize)).
      children(
        arg[String]("<partitions>").
          action((x, c) => c.copy(partitionPath = x)).
          text("The partition model."),

        arg[String]("<source>").
          action((x, c) => c.copy(sourcePath = x)).
          text("Source of samples in CSV format."),

        arg[String]("<output>").
          action((x, c) => c.copy(outputPath = x)).
          text("Location to save the KML file.")
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
          case 'visualize => visualize(sc, arguments)
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

  def visualize(sc: SparkContext, args: Arguments) = {
    val rand = new Random()
    def randomColor() = {
      val bytes = Array.ofDim[Byte](3)
      rand.nextBytes(bytes)
      f"ff${bytes(0)}%02x${bytes(1)}%02x${bytes(2)}%02x"
    }

    val model = KMeansPartitioner.load(sc, args.partitionPath)
    val samples = loadSamples(sc, args.sourcePath)
    val predictions = samples.zip(model.partition(samples))

    val kml =
    <kml><Document>
      {for (i <- 0 until model.numPartitions) yield
        <Style id={i.toString}><IconStyle>
        <color>{randomColor()}</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle></Style>
      } {for (m <- predictions.toLocalIterator) yield
        <Placemark>
        <description>{m._1.tags.toString}</description>
        <styleUrl>{m._2.toString}</styleUrl>
        <Point><coordinates>{s"${m._1.longitude},${m._1.latitude}"}</coordinates></Point>
        </Placemark>
    } </Document></kml>
    
    scala.xml.XML.save(args.outputPath, kml)
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
