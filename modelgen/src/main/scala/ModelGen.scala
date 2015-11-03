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
  modelPath: String = null,
  outputPath: String = null,
  tags: String = null,
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

    note("\n")
    cmd("train").
      text("Train the Naive Bayes classification model.").
      action((_, c) => c.copy(command = 'train)).
      children(
        arg[String]("<partitions>").
          action((x, c) => c.copy(partitionPath = x)).
          text("The partition model."),

        arg[String]("<source>").
          action((x, c) => c.copy(sourcePath = x)).
          text("Source of samples in CSV format."),

        arg[String]("<output>").
          action((x, c) => c.copy(modelPath = x)).
          text("Location to save the generated model.")
      )

    note("\n")
    cmd("label").
      text("Write a CSV file with classes assigned to each sample.").
      action((_, c) => c.copy(command = 'label)).
      children(
        arg[String]("<partitions>").
          action((x, c) => c.copy(partitionPath = x)).
          text("The partition model."),

        arg[String]("<source>").
          action((x, c) => c.copy(sourcePath = x)).
          text("Source of samples in CSV format."),

        arg[String]("<output>").
          action((x, c) => c.copy(outputPath = x)).
          text("Location to save the generated model.")
      )

    note("\n")
    cmd("predict").
      text("Predict the location of a set of tags.").
      action((_, c) => c.copy(command = 'predict)).
      children(
        arg[String]("<partitions>").
          action((x, c) => c.copy(partitionPath = x)).
          text("The partition model."),

        arg[String]("<model>").
          action((x, c) => c.copy(modelPath = x)).
          text("The classification model."),

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
          case 'partition => partition(sc, arguments)
          case 'visualize => visualize(sc, arguments)
          case 'train => train(sc, arguments)
          case 'label => label(sc, arguments)
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

  def train(sc: SparkContext, args: Arguments) = {
    val samples = loadSamples(sc, args.sourcePath)
    val partitioner = KMeansPartitioner.load(sc, args.partitionPath)
    val model = TagClassifier.train(samples, partitioner)
    model.save(sc, args.modelPath)
  }

  def label(sc: SparkContext, args: Arguments) = {
    val samples = loadSamples(sc, args.sourcePath)
    val partitioner = KMeansPartitioner.load(sc, args.partitionPath)
    val labeled = samples.zip(partitioner.partition(samples))

    val writer = new PrintWriter(args.outputPath)
    try {
      val format = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
      for (m <- labeled.toLocalIterator) {
        writer.println(s"""${m._2},${m._1.id},${m._1.userId},${m._1.date.format(format)},"{${m._1.tags.mkString(",")}}",${m._1.locationId},${m._1.locationName},${m._1.latitude},${m._1.longitude}""")
      }
    } finally {
      writer.close()
    }
  }

  def predict(sc: SparkContext, args: Arguments) = {
    val partitioner = KMeansPartitioner.load(sc, args.partitionPath)
    val classifier = TagClassifier.load(sc, args.modelPath)
    val tags = TagList.parse(args.tags)
    val clas = classifier.predict(tags)
    val location = partitioner.partitionCenter(clas)
    println(s"PREDICTION: ${location.toString}")
  }

  def loadSamples(sc: SparkContext, path: String) = {
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
      }).
      cache()
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
