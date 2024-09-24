import numpy as np


class CurveGenerator:
    def __init__(self, tension=0.5):
        self.points = []
        self.curves = []
        self.tension = tension
        self.pathPoses = []

    def hermiteCurve(self, p0, p1, t0, t1):
        distance = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        numPoints = int(distance // 0.1)

        t = np.linspace(0, 1, numPoints)
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
        y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]
        return x.tolist(), y.tolist()
            
    def createTangentPoints(self):
        self.tangentPoints = []
        if len(self.points) < 2:
            return

        for i in range(len(self.points)):
            if i == 0:
                diffX = self.points[1][0] - self.points[0][0]
                diffY = self.points[1][1] - self.points[0][1]
            elif i == len(self.points) - 1:
                diffX = self.points[i][0] - self.points[i - 1][0]
                diffY = self.points[i][1] - self.points[i - 1][1]
            else:
                diffX = self.points[i + 1][0] - self.points[i - 1][0]
                diffY = self.points[i + 1][1] - self.points[i - 1][1]
            self.tangentPoints.append(((1 - self.tension) * diffX, (1 - self.tension) * diffY))

    def createCurve(self):
        self.curves = []
        if len(self.points) < 2:
            return

        self.createTangentPoints()

        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            t0 = self.tangentPoints[i]
            p1 = self.points[i + 1]
            t1 = self.tangentPoints[i + 1]

            curveX, curveY = self.hermiteCurve(p0, p1, t0, t1)
            self.curves.append((curveX, curveY))
        
    def wayPoints(self):
        if len(self.points) < 2:
            return
        
        self.createCurve()
        distanceTraced = 0
        prevX, prevY = 0, 0
        self.pathPoses = []  # Initialize the path poses

        for idx, curve in enumerate(self.curves):
            curveX, curveY = curve
            for i in range(len(curveX)):
                x, y = curveX[i], curveY[i]
                
                if i < len(curveX) - 1:
                    nextX, nextY = curveX[i+1], curveY[i+1]
                else:
                    if idx < len(self.curves) - 1:
                        nextCurve = self.curves[idx + 1]
                        nextX, nextY = nextCurve[0][0], nextCurve[1][0]
                    else:
                        break

                distanceTraced += np.sqrt((x - prevX) ** 2 + (y - prevY) ** 2)
                yaw = np.arctan2(nextY - y, nextX - x)
                self.pathPoses.append([x, y, yaw, distanceTraced])
                prevX, prevY = x, y