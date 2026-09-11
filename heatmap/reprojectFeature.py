import pyproj
from shapely.ops import transform

def reprojectPolygon(polygon, inCrs='epsg:4326', outCrs='epsg:3857'):
  """Reproject Polygon/MultiPolygon to a different coordinate system.

  Args:
    polygon: accepts a Shapely Polygon or MultiPolygon
    inCrs: coordinate system that input polygon is in, defaults to ep
    outCrs: coordinate system to reproject to, as epsg string.  Defaults to 'epsg:3857'
  """
  # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe

  project = pyproj.Transformer.from_proj(
    pyproj.Proj(inCrs), # source coordinate system
    pyproj.Proj(outCrs), # destination coordinate system
    always_xy=True)
  
  reprojected = transform(project.transform, polygon)  # apply projection
  return reprojected
