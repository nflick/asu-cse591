lazy val root = (project in file(".")).
  settings(
    name := "ModelGen",
    version := "0.1-SNAPSHOT",
    scalaVersion := "2.10.4",
    scalacOptions ++= Seq("-feature"),
    
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "1.5.1" % "provided",
      "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided",
      "com.github.scopt" %% "scopt" % "3.3.0",
      "org.scalatest" % "scalatest_2.10" % "2.0" % "test"
    ),

    resolvers ++= Seq(
      Resolver.sonatypeRepo("public"),
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    )
  )
