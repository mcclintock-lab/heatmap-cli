import math
import os
import numpy as np
import rasterio
from rasterio.features import bounds, rasterize
from rasterio.enums import MergeAlg
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import shape, box, Polygon, MultiPolygon
import fiona
import simplejson
import time
import datetime
from .reprojectFeature import reprojectPolygon
from .calc_raster_props import calcRasterProps
from .calc_sap import calcSap
from .shapeIndex import calcShapeIndex, calcPolygonShapeIndex
from .progress import FeatureProgress, progress_write

def countRasterCells(geometry, transform, width, height, all_touched=False):
  """Count output raster cells a geometry would cover under the given rasterize mode.

  Uses a window clipped to the geometry bounds (padded by 1 pixel for ALL_TOUCHED
  edges) so counting does not allocate a full-size raster per shape.
  """
  geom = shape(geometry) if isinstance(geometry, dict) else geometry
  minx, miny, maxx, maxy = geom.bounds
  window = from_bounds(minx, miny, maxx, maxy, transform)

  row_off = int(math.floor(window.row_off)) - 1
  col_off = int(math.floor(window.col_off)) - 1
  row_end = int(math.ceil(window.row_off + window.height)) + 1
  col_end = int(math.ceil(window.col_off + window.width)) + 1

  row_off = max(0, row_off)
  col_off = max(0, col_off)
  row_end = min(height, row_end)
  col_end = min(width, col_end)

  win_height = row_end - row_off
  win_width = col_end - col_off
  if win_height <= 0 or win_width <= 0:
    return 0

  window_transform = transform * Affine.translation(col_off, row_off)
  mask = rasterize(
    [(geom, 1)],
    out_shape=(win_height, win_width),
    transform=window_transform,
    fill=0,
    all_touched=all_touched,
    dtype='uint8'
  )
  return int(mask.sum())


def countInfileFeatures(infile):
  """Return the feature count of a vector file, or 0 if it cannot be opened."""
  if not infile:
    return 0
  try:
    with fiona.open(infile) as src:
      return len(src)
  except (fiona.errors.DriverError, OSError, TypeError, ValueError):
    return 0


def genHeatMap(
  infile,
  outPath=None,
  overwrite=False,
  method='sap',
  importanceField=None,
  importanceFactorField=None,
  areaFactor=1,
  uniqueIdField=None,
  outCrsString='epsg:3857',
  outResolution=1000,
  bounds=None,
  boundsPrecision=0,
  allTouchedSmall=False,
  allTouchedSmallFactor=1.25,
  fixGeom=False,
  maxArea=None,
  maxSap=None,
  areaFloor=None,
  logToFile=False,
  progress=None,
):
  """Generates Spatial Access Priority (SAP) raster map given run configuration

  Arguments:
    infile: path+filename of vector dataset containing features, format must be supported by fiona/gdal
    outpath: path to output heatmaps to.  Filename will be the same as the input, just with the .tif extension.  If not specified, heatmaps are output to the input folder
    overwrite: whether to overwrite existing heatmap output, defaults to false and skips
    method: method for calculating value: count, area, sap. Defaults to sap
    importanceField: name of vector attribute containing importance value used for SAP calculation
    importanceFactorField: name of vector attribute containing importanceFactor value for importance
    areaFactor: factor to change the area by dividing. Area is overlapped raster cell coverage (cell count × cell area). For example if cell area is in square meters, an areaFactor of 1,000,000 will make the SAP per square km. because 1 sq. km = 1000m x 1000m = 1mil sq. meters 
    uniqueIdField: field containing a unique Id for feature to use for logging the list of features included in the raster for verification.  Must not allow person to be re-identified
    outCrsString: the epsg code for the output raster coordinate system, defaults to epsg:3857 aka Web Mercator
    outResolution: length/width of planning unit in units of output coordinate system, defaults to 1000 (1000m = 1km)
    bounds: bounds to use for output raster, as [w, s, e, n] in CRS of infile.  Output raster will align to the top left, but will extend past the bottom right as needed to the next multiple of outResolution
    boundsPrecision: number of digits to round the coordinates of bound calculation to. useful if don't snap to numbers as expected
    allTouchedSmall: (boolean) use allTouched rasterize option for shapes with smaller shape index than a raster cell (area/perimeter length).  Ensures small and narrow shapes are not lost and every shape contributes heat to at least one pixel in result. Larger shapes are still picked up using Bresenham’s line algorithm because allTouched creates some seemingly invalid output (double counting) along shape boundaries. Using allTouched only for smallest shapes that need it mitigates this, but also uses additional memory. SAP and area methods always use overlapped raster cell area (cell count × cell area) rather than vector geometry area, counting cells with the same allTouched setting used to burn the shape.
    allTouchedSmallFactor: (number) use to increase the shapeIndex threshold for identifying small shapes.  shapeIndex threshold is calculated as (shapeIndex of a raster cell * allTouchedSmallFactor).  Defaults to 1.25.  Increasing the factor will identify increasingly larger shapes as "small" and to be run with AllTouched option.  Useful when you have polygons that are mostly large but have small areas that are long and narrow and thus spotty in being picked up
    fixGeom: if an invalid geometry is found, if fixGeom is True it attempts to fix using buffer(0), otherwise it fails.  Review the log to make sure the automated fix was acceptable
    maxArea: limits the area of a shape in SAP calculation after areaFactor. Gives shapes with high area an artificially lower one, increasing their SAP
    areaFloor: minimum overlapped cell area in square meters. Calculated areas below this are raised to the floor before areaFactor, decreasing their SAP. Does not affect allTouchedSmall classification, which uses vector shape index only.
    maxSap: limits the SAP value. Gives shapes with high priority an artificially lower one, decreasing their presence in heatmap
    logToFile: (boolean) whether to write the .log.txt and .error.geojson and skip printing the full info to stdout. The .info.json is always written to a logs/ directory alongside the output GeoTIFF. When logToFile is True, the log and error files are written to the same directory, overwriting any existing files with the same name.
    progress: optional FeatureProgress tracker. When running multiple files from the CLI, a shared tracker is passed so feature progress is counted against all vector files. If omitted, a bar is created for this run only.

  Returns:
    Manifest of run
  """
  startTime = time.perf_counter()
  
  try:
    src_shapes = fiona.open(infile)
  except (fiona.errors.DriverError):
    progress_write(progress, 'Warning: infile not found, skipping {0}'.format(infile))
    return None

  if len(src_shapes) < 1:
    progress_write(progress, 'Warning: infile contains no features, skipping {0}'.format(infile))
    return None
  
  outCrs = CRS.from_string(outCrsString)
  error_shapes = []

  # output files have the same name as infile
  inpath, inFullFilename = os.path.split(infile)
  inFilename = inFullFilename.split('.')[0]
  outDir = inpath if outPath is None else outPath
  if outPath is not None:
    os.makedirs(outDir, exist_ok=True)
  # full path, minus extension
  inBasename = os.path.join(outDir, inFilename)

  outfile = "{}.tif".format(inBasename)
  outfileSmall = "{}_small.tif".format(inBasename)
  outfileLarge = "{}_large.tif".format(inBasename)

  logsDir = os.path.join(outDir, 'logs')
  logBase = os.path.join(logsDir, inFilename)
  infofile = "{}.info.json".format(logBase)
  logfile = "{}.log.txt".format(logBase) if logToFile else None
  errorfile = "{}.error.geojson".format(logBase) if logToFile else None

  if os.path.exists(outfile) and not overwrite:
    progress_write(progress, 'Warning: outfile {0} already exists, skipping. Remove it and re-run or use overwrite option'.format(outfile))
    if progress is not None:
      progress.update(len(src_shapes))
    return None

  manifest = {
    'timestamp': datetime.datetime.now().astimezone().isoformat(),
    'params': {
      'infile': infile,
      'outfile': outfile,
      'logfile': logfile,
      'infofile': infofile,
      'errorfile': errorfile,
      'importanceField': importanceField,
      'importanceFactorField': importanceFactorField,
      'uniqueIdField': uniqueIdField,
      'outCrsString': outCrsString,
      'outResolution': outResolution,
      'bounds': bounds,
      'boundsPrecision': boundsPrecision,
      'allTouchedSmall': allTouchedSmall,
      'areaFactor': areaFactor,
      'maxArea': maxArea,
      'areaFloor': areaFloor,
      'maxSap': maxSap,
    },
    'included': [],
    'includedSmall': [],
    'excluded': [],
    'fixed': [],
    'includedCount': 0,
    'excludedCount': 0,
    'includedSmallCount': 0
  }
  log = []

  inBounds = bounds if bounds else src_shapes.bounds
  (outBounds, width, height, outTransform) = calcRasterProps(inBounds, src_shapes.crs, outCrsString, outResolution, boundsPrecision)  

  manifest['height'] = height
  manifest['width'] = width
  manifest['inBounds'] = inBounds
  manifest['outBounds'] = outBounds

  # Generate a list of tuples, each consisting of the geometry and importance, as expected by rasterize
  shapes = []
  # Special handle shapes smaller than an output pixel
  smallShapes = []
  # Parallel to `included`: (uniqueId, heatValue) for ALL_TOUCHED shapes
  smallIncluded = []

  # Calculate shape index of one raster pixel.  Shape index = pixel area / pixel length
  # Can be used to identify small shapes that might not be picked up by the standard
  # rasterize function which uses painters algorithm (shape must cross centerpoint of raster cell)
  shapeIndexThreshold = calcShapeIndex(inBounds, outResolution, allTouchedSmallFactor)
  cellArea = outResolution ** 2

  def heatValueFromCellCount(nCells):
    rasterArea = max(nCells, 1) * cellArea
    if method == 'area':
      if areaFloor:
        rasterArea = max(rasterArea, areaFloor)
      return 1 / rasterArea
    elif method == 'sap':
      return calcSap(
          shapeGeom,
          feature['properties'][importanceField] if importanceField else 1,
          areaFactor,
          feature['properties'][importanceFactorField] if importanceFactorField else 1,
          maxArea,
          maxSap,
          area=rasterArea,
          areaFloor=areaFloor
        )
    return 1 # count method

  owns_progress = progress is None
  if owns_progress:
    progress = FeatureProgress(len(src_shapes))
  progress.start(inFilename)

  try:
    for idx, feature in enumerate(src_shapes):
      # Convert to shapely Polygon/MultiPolygon with reproject if necessary
      shapeGeom = shape(feature['geometry']) if src_shapes.crs['init'] == outCrsString else reprojectPolygon(shape(feature['geometry']), src_shapes.crs['init'], outCrsString)
      # Get new geojson-like object from shape, we'll use it for lower level work later
      geometry = shapeGeom.__geo_interface__

      uniqueId = ''
      if uniqueIdField:
        uniqueId = feature['properties'][uniqueIdField]
      else:
        uniqueId = idx

      error = False
      heatValue = None
      # If shape is invalid, attempt to fix it, otherwise log it and move on
      if not shapeGeom.is_valid:
        if fixGeom:
          fixedGeom = shapeGeom.buffer(0)
          if fixedGeom.is_valid and fixedGeom.area > 0:
            log.append("Fixed invalid feature geometry")
            log.append(simplejson.dumps(feature))
            log.append("With new geometry")
            logGeom = reprojectPolygon(fixedGeom.__geo_interface__, "epsg:3857", "epsg:4326")
            log.append(simplejson.dumps({
              **feature,
              'geometry': logGeom
            }))
            log.append("")     
            shapeGeom = fixedGeom
            geometry = shapeGeom.__geo_interface__
            if uniqueIdField:
              manifest['fixed'].append(feature['properties'][uniqueIdField])
            else:
              manifest['fixed'].append(idx + 1)
          else:
            error = "Geometry is invalid or area is 0, attempted fix failed" 
            error_shapes.append(feature)
        else:
            error = "Geometry is invalid"
            error_shapes.append(feature)
      elif shapeGeom.area == 0:
        error = "Area of geometry is zero" 
      elif len(geometry['coordinates'][0]) == 0:
        error = "Geometry has no coordinates"

      # Classify small vs large, count overlapped cells, then calculate heat from cell area
      if not error:
        # Split shapes into two groups based on whether their shape index is above or below threshold
        # Threshold is based on the size of one raster cell
        # Shapes below threshold are considered small and will have the all_touched algorithm applied
        # to ensure each small shape gets at least one pixel of heat in the result
        # Regular shapes will not have all_touched algorithm applied because all_touched causes some
        # line artifacts for larger shapes that are incorrect. Using only on small minimizes that affect

        # Classify from vector geometry only (area / perimeter). Cell-area SAP and
        # areaFloor are applied after this and must not change isSmall.
        if (geometry["type"] == "Polygon"):
          isSmall = allTouchedSmall and calcPolygonShapeIndex(shapeGeom) < shapeIndexThreshold
          nCells = countRasterCells(shapeGeom, outTransform, width, height, all_touched=isSmall)
          heatValue = heatValueFromCellCount(nCells)
          if (isSmall):
            smallShapes.append((geometry, heatValue))
            smallIncluded.append((uniqueId, heatValue))
          else:
            shapes.append((geometry, heatValue))
        elif (geometry["type"] == "MultiPolygon"):
          # If multipolygon, split up into individual polygons so that small pieces bin accordingly
          # One SAP per feature: sum cell counts across parts, then apply that heat value to each
          polys = list(shapeGeom.geoms)
          parts = []
          totalN = 0
          for p in polys:
            isSmall = allTouchedSmall and calcPolygonShapeIndex(p) < shapeIndexThreshold
            totalN += countRasterCells(p, outTransform, width, height, all_touched=isSmall)
            parts.append((p, isSmall))
          heatValue = heatValueFromCellCount(totalN)
          for p, isSmall in parts:
            if (isSmall):
              smallShapes.append((MultiPolygon([p]).__geo_interface__, heatValue))
              smallIncluded.append((uniqueId, heatValue))
            else:
              shapes.append((p.__geo_interface__, heatValue))

        manifest['included'].append((uniqueId, heatValue))
      elif len(error) > 0:
        log.append("Skipping feature: {}".format(error))
        log.append(simplejson.dumps(feature))
        log.append("")
        manifest['excluded'].append((uniqueId, heatValue))

      progress.update(1)

  finally:
    progress.finish()

  n_small = len(smallShapes) if (allTouchedSmall and len(smallShapes) > 0) else 0
  n_large = len(shapes)
  result = None
  if n_small > 0 or n_large > 0:
    progress_write(progress, 'Rasterizing {} ({} geoms)'.format(inFilename, n_small + n_large))
    result = np.zeros((height, width), dtype='float32')
    if n_small > 0:
      rasterize(
        smallShapes,
        out=result,
        transform=outTransform,
        merge_alg=MergeAlg.add,
        fill=0,
        all_touched=True
      )
    if n_large > 0:
      rasterize(
        shapes,
        out=result,
        transform=outTransform,
        merge_alg=MergeAlg.add,
        fill=0,
        all_touched=False
      )

  os.makedirs(logsDir, exist_ok=True)

  if logfile:
    with open(logfile, 'w') as logFile:
      for item in log:
          logFile.write("%s\n" % item)

  if result is None:
    result = np.zeros((height, width), dtype='float32')
  progress_write(progress, 'Writing {}'.format(outfile))
  with rasterio.open(
    outfile,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    nodata=0,
    dtype='float32',
    crs=outCrs,
    transform=outTransform,
    compress='deflate'
  ) as out:
    out.write(result, indexes=1)

  if not logfile and len(log) > 0:
      progress_write(progress, 'Log:')
      for item in log:
        progress_write(progress, item)
      progress_write(progress, '')

  manifest['included'] = [(uid, hv) for uid, hv in manifest['included']]
  manifest['includedSmall'] = [(uid, hv) for uid, hv in smallIncluded]
  manifest['includedCount'] = len(manifest['included'])
  manifest['excludedCount'] = len(manifest['excluded'])
  manifest['includedSmallCount'] = len(smallIncluded)
  manifest['executionTime'] = round(time.perf_counter() - startTime, 2)
  if allTouchedSmall:
    manifest['allTouchedSmallFactor'] = allTouchedSmallFactor
    manifest['shapeIndexThreshold'] = shapeIndexThreshold

  progress_write(progress, 'Created SAP raster {} in {}s'.format(outfile, manifest['executionTime']))

  progress_write(progress, ' {} features burned in'.format(manifest['includedCount']))
  if (allTouchedSmall):
    progress_write(progress, ' allTouchedSmall enabled, numSmallShapes: {0}'.format(len(smallShapes)))
  if manifest['excludedCount'] > 0:
    progress_write(progress, ' {} features excluded, see logfile for details'.format(manifest['excludedCount']))
  progress_write(progress, '')

  with open(infofile, 'w') as infoFile:
    simplejson.dump(manifest, infoFile, indent=2)

  if errorfile and len(error_shapes) > 0:
    with open(errorfile, 'w') as errorFile:
      errorFile.write(simplejson.dumps({
        "type": "FeatureCollection",
        "features": error_shapes
      }))

  return manifest
  


  
