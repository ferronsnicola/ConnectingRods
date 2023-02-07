import math
import utils
import cv2

class Hole:

    def __init__(self, points, contour): # nella versione piu efficiente, si e' scelto di calcolare i diametri sfruttando il MinEnclosingCircle, quindi points risulta un parametro inutile
                                         # si e' tuttavia deciso di lasciarlo per completezza ed eventuali test
        #self.points = points
        self.contour = contour
        self.center, self.diameter = self.compute_mec()
        #self.center = utils.compute_barycenter(points)
        #self.diameter = self.compute_diameter_by_contour()

    def compute_mec(self):
        (x, y), radius = cv2.minEnclosingCircle(self.contour)
        center = (int(round(x)), int(round(y)))
        radius = int(round(radius)) - 1 # il -1 e' dovuto al fatto che voglio coincidenza invece che tangenza
        return center, 2*radius

    def compute_diameter_by_area(self):
        area = len(self.points)
        r = math.sqrt(area/math.pi)
        return 2*r

    def compute_diameter_by_contour(self):
        r=0
        for i in range(len(self.contour)):
            r += utils.distance(self.contour[i][0], self.center)
        return 2*r/len(self.contour)

    def __str__(self):
        return "center=" + str(self.center) + ", diameter=" + str(self.diameter)
