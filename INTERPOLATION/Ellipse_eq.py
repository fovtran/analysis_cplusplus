from math import *

def bisect(f,t_0,t_1,err=0.0001,max_iter=100):
    iter = 0
    ft_0 = f(t_0)
    ft_1 = f(t_1)
    assert ft_0*ft_1 <= 0.0
    while True:
        t = 0.5*(t_0+t_1)
        ft = f(t)
        if iter>=max_iter or ft<err:
            return t
        if ft * ft_0 <= 0.0:
            t_1 = t
            ft_1 = ft
        else:
            t_0 = t
            ft_0 = ft
        iter += 1

class Ellipse(object):
    def __init__(self,center,radius,angle=0.0):
        assert len(center) == 2
        assert len(radius) == 2
        self.center = center
        self.radius = radius
        self.angle = angle

    def distance_from_origin(self):
        """                                                                           
        Ellipse equation:                                                             
        (x-center_x)^2/radius_x^2 + (y-center_y)^2/radius_y^2 = 1                     
        x = center_x + radius_x * cos(t)                                              
        y = center_y + radius_y * sin(t)                                              
        """
        center = self.center
        radius = self.radius

        # rotate ellipse of -angle to become axis aligned                             

        c,s = cos(self.angle),sin(self.angle)
        center = (c * center[0] + s * center[1],
                  -s* center[0] + c * center[1])

        f = lambda t: (radius[1]*(center[1] + radius[1]*sin(t))*cos(t) -
                       radius[0]*(center[0] + radius[0]*cos(t))*sin(t))

        if center[0] > 0.0:
            if center[1] > 0.0:
                t_0, t_1 = -pi, -pi/2
            else:
                t_0, t_1 = pi/2, pi
        else:
            if center[1] > 0.0:
                t_0, t_1 = -pi/2, 0
            else:
                t_0, t_1 = 0, pi/2

        t = bisect(f,t_0,t_1)
        x = center[0] + radius[0]*cos(t)
        y = center[1] + radius[1]*sin(t)
        return sqrt(x**2 + y**2)

print Ellipse((1.0,-1.0),(2.0,0.5)).distance_from_origin()