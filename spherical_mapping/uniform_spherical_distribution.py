# Function to distribute N points on the surface of a sphere
# (source: http://www.softimageblog.com/archives/115)
def uniform_spherical_distribution(N):
    pts = []
    inc = math.pi * (3 - math.sqrt(5))
    off = 2 / float(N)
    for k in range(0, int(N)):
        y = k * off - 1 + (off / 2)
        r = math.sqrt(1 - y*y) 
        phi = k * inc
        pts.append([math.cos(phi)*r, y, math.sin(phi)*r])
    return pts
