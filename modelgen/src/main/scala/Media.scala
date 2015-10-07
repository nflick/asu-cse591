/**
 * ModelGen.scala
 * Representations for Instagram media with geolocations.
 * Author: Nathan Flick
 */

case class Media(
  id: Long,
  tags: String,
  latitude: Double,
  longitude: Double
)