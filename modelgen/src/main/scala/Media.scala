/**
 * ModelGen.scala
 * Representations for Instagram media with geolocations.
 * Author: Nathan Flick
 */

import java.time.LocalDateTime

case class Media(
  id: Long,
  userId: Long,
  date: LocalDateTime,
  tags: Array[String],
  locationId: Long,
  locationName: String,
  latitude: Double,
  longitude: Double
)
