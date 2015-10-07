lazy val root = (project in file(".")).
  settings(
    name := "ModelGen",
    version := "0.1-SNAPSHOT",
    scalaVersion := "2.10.4",
    scalacOptions ++= Seq("-feature"),
    libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "1.5.1",
      "org.apache.spark" %% "spark-mllib" % "1.5.1")
  )