from numpy import inf
from shapely.geometry import Polygon

def calcShapeIndex(bounds, resolution, factor=1.25):
  """Given parameters of a raster and optional factor, returns a threshold for determining if a shape is smaller than one raster cell (pixel)
  Returns threshold that determines if a shape , and the `cellShapeIndex` that was determined from the raster params
  Args:
    bounds: the bounding box of raster
    resolution: the cell resolution of raster
    factor: 
  """ 
  # Use shape index for identifying "small" shapes
  # Based on https://gis.stackexchange.com/questions/316128/identifying-long-and-narrow-polygons-in-with-postgis
  # In practice, the inverse method does not seem to work as well
  
  # Calculate shape index for one raster pixel
  cellShapeIndex = calcCellShapeIndex(bounds, resolution)
  threshold = inf
  threshold = cellShapeIndex * factor
  return threshold


def calcCellShapeIndex(bounds, resolution):
  """Given raster bounds and resolution, calculates the shape index of one raster cell/pixel
  Shape index = pixel area / pixel length
  """
  # Get shape index for one output raster cell
  (minx, miny, maxx, maxy) = bounds
  cellBL = (minx, miny)
  cellBR = (minx + resolution, miny)
  cellTR = (minx + resolution, miny + resolution)
  cellTL = (minx, miny + resolution)
  cellPoly = Polygon([cellBL, cellBR, cellTR, cellTL, cellBL])
  # Calculate shape index for polygon
  cellShapeIndex = cellPoly.area / cellPoly.length
  return cellShapeIndex


def calcPolygonShapeIndex(polygon):
  """Returns shape index of given polygon

  Args:
    polygon: accepts a Shapely Polygon or MultiPolygon
  """
  curLength = 1
  if (polygon.__geo_interface__["type"] == "Polygon"):
    curLength = polygon.exterior.length
  elif (polygon.__geo_interface__["type"] == "MultiPolygon") :
    curLength = polygon.length
  shapeIndex = polygon.area / curLength

  return shapeIndex