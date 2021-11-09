# create uniform spiral grid
numOfPoints = varargin[0]
vxyz = zeros((numOfPoints,3),dtype=float)
sq0 = 0.00033333333**2
sq2 = 0.9999998**2
sumsq = 2*sq0 + sq2
vxyz[numOfPoints -1] = array([(sqrt(sq0/sumsq)),
                              (sqrt(sq0/sumsq)),
                              (-sqrt(sq2/sumsq))])
vxyz[0] = -vxyz[numOfPoints -1]
phi2 = sqrt(5)*0.5 + 2.5
rootCnt = sqrt(numOfPoints)
prevLongitude = 0
for index in arange(1, (numOfPoints -1), 1, dtype=float):
  zInc = (2*index)/(numOfPoints) -1
  radius = sqrt(1-zInc**2)

  longitude = phi2/(rootCnt*radius)
  longitude = longitude + prevLongitude
  while (longitude > 2*pi):
    longitude = longitude - 2*pi

  prevLongitude = longitude
  if (longitude > pi):
    longitude = longitude - 2*pi

  latitude = arccos(zInc) - pi/2
  vxyz[index] = array([ (cos(latitude) * cos(longitude)) ,
                        (cos(latitude) * sin(longitude)),
                        sin(latitude)])
