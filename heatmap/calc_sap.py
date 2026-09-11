def calcSap(geometry, importance=1, areaFactor=1, importanceFactor=1, maxArea=None, maxSap=None, area=None, areaFloor=None):
  """Core algorithm, calculates the spatial access priority (SAP) value for a single area

  Each respondent has a total SAP of 100, and each shape is assigned a portion
  of that 100. The SAP for a shape is calculated as (Importance / Area), which
  can be interpreted as "importance per area unit". Area is the overlapped
  raster cell coverage (cell count × cell area) when provided by the caller,
  otherwise the geometry area in the coordinate system used. For Web Mercator
  that yields square meters. Use the areaFactor option to change this.
  
  Use the optional importanceFactor to accommodate additional variables, changing what the SAP represents.

  Parameters:
    geometry: Shapely geometry in 3857 (unit of meters)
    importance: importance of geometry (1-100)
    areaFactor: factor to change the area by dividing. For example if area of geometry is calculated in square meters, an areaFactor of 1,000,000 will make the SAP per square km. because 1 sq. km = 1000m x 1000m = 1mil sq. meters
    importanceFactor: numeric value to multiply the importance by.  Use to scale the SAP from being 'per respondent' to a larger group of livelihoods or even economic values 
    maxArea: limits the area of a shape in SAP calculation.  Gives shapes with high area an artifically lower one, increasing their SAP relative to others, increasing their presence in heatmap
    maxSap: limits the SAP value. Gives shapes with high priority an artificially lower one, decreasing their presence in heatmap
    area: overlapped raster cell area in the same units as geometry.area.  When omitted, geometry.area is used
    areaFloor: minimum area in square meters (same units as geometry.area / cell area).  Calculated areas below this are raised to the floor before areaFactor, decreasing their SAP.  Does not change the geometry and is not used to classify small shapes.
  Returns:
    SAP value for shape
  """
  
  area = geometry.area if area is None else area

  if (areaFloor):
    area = max(area, areaFloor)

  area = area / areaFactor
  
  if (maxArea):
    area = min(area, maxArea)

  sap = importanceFactor * importance / area

  if (maxSap):
    sap = min(sap, maxSap)
  
  return sap
