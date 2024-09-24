import numpy as np
import matplotlib.pyplot as plt
from Polynomial import Polynomial
import time
import copy

ROBOT_RADIUS = 0.33

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


class FrenetPath:
    def __init__(self):
        self.predictionTime = []      # Prediction Time

        self.latPos = []       # Longitudinal Data
        self.latVel = []       # Longitudinal Data
        self.latAcc = []       # Longitudinal Data
        self.latJerk = []       # Longitudinal Data

        self.lonPos = []       # Longitudinal Data
        self.lonVel = []       # Longitudinal Data
        self.lonAcc = []       # Longitudinal Data
        self.lonJerk = []       # Longitudinal Data

        self.worldX = []
        self.worldY = []
        self.worldYaw = []
        self.worldD = []

        self.cd = 0.0
        self.cs = 0.0
        self.costFunction = 0.0     # Cost

        self.maxCurvature = []


class OptimalTrajectoryPlanner:
    def __init__(self, points):
        self.generator = CurveGenerator()
        self.generator.points = points

        self.maxVelocity_ = 1.5
        self.maxAcceleration_ = 0.6
        self.maxCurvature_ = 1.0

        self.MAX_ROAD_WIDTH_ = ROBOT_RADIUS * 6.0
        self.D_ROAD_W_ = ROBOT_RADIUS * 2
        self.DT_ = 0.4125

        self.MAX_T_ = ROBOT_RADIUS * 8.0 * 2
        self.MIN_T_ = ROBOT_RADIUS * 6.0 * 2

        self.targetVelocity_ = 0.8
        self.D_T_S_ = 0.05

        self.klat_ = 1
        self.klon_ = 1

        self.kjd_ = 0.1
        self.ktd_ = 0.1
        self.ksd_ = 1

        self.kjs_ = 0.1
        self.kts_ = 0.1
        self.kss_ = 1

    def frenetTrajectory(self, d0, dv0, da0, s0, sv0, sa0):
        frenetPaths = []

        for di in np.arange(-self.MAX_ROAD_WIDTH_, self.MAX_ROAD_WIDTH_, self.D_ROAD_W_):
            for Ti in np.arange(self.MIN_T_, self.MAX_T_, self.DT_):
                
                path = FrenetPath()
                quintic = Polynomial(d0, dv0, da0, di, 0, 0, Ti)

                path.predictionTime = [t for t in np.arange(0, Ti , self.DT_)]

                path.latPos = [quintic.position(t) for t in path.predictionTime]
                path.latVel = [quintic.velocity(t) for t in path.predictionTime]
                path.latAcc = [quintic.acceleration(t) for t in path.predictionTime]
                path.latJerk = [quintic.jerk(t) for t in path.predictionTime]

                for tv in np.arange(self.targetVelocity_ - self.D_T_S_, 
                                    self.targetVelocity_ + self.D_T_S_, 
                                    self.D_T_S_):
                    
                    fPath = copy.deepcopy(path)
                    quartic = Polynomial(s0, sv0, sa0, xT=None, vT=tv, aT=0, T=Ti)

                    fPath.lonPos = [quartic.position(t) for t in path.predictionTime]
                    fPath.lonVel = [quartic.velocity(t) for t in path.predictionTime]
                    fPath.lonAcc = [quartic.acceleration(t) for t in path.predictionTime]
                    fPath.lonJerk = [quartic.jerk(t) for t in path.predictionTime]

                    jd = sum(np.power(fPath.latJerk, 2))
                    js = sum(np.power(fPath.lonJerk, 2))


                    fPath.cd = jd * self.kjd_ + Ti * self.ktd_ + np.power(fPath.latPos[-1], 2) * self.ksd_
                    fPath.cs = js * self.kjs_ + Ti * self.kts_ + np.power(self.targetVelocity_ - fPath.lonVel[-1], 2) * self.kss_
                    fPath.costFunction = self.klat_ * fPath.cd + self.klon_ * fPath.cs
    
                    frenetPaths.append(fPath)

        return frenetPaths


    def globalCoordinates(self, frenetPaths, globalPath):
        for path in frenetPaths:
            j = 0
            for i in range(len(path.lonPos)):
                x, y, yaw = 0, 0, 0
                while j < len(globalPath) and abs(path.lonPos[i] - globalPath[j][3]) > 0.1:
                    j += 1
                if j < len(globalPath):
                    x = globalPath[j][0]
                    y = globalPath[j][1]
                    yaw = globalPath[j][2]
                d = path.latPos[i]

                path.worldX.append(x + d * np.cos(yaw + 1.57))
                path.worldY.append(y + d * np.sin(yaw + 1.57))

            for i in range(len(path.worldX) - 1):
                dx = path.worldX[i + 1] - path.worldX[i]
                dy = path.worldY[i + 1] - path.worldY[i]
                distance = np.sqrt(dx**2 + dy**2)
                
                path.worldYaw.append(np.arctan2(dy, dx))
                path.worldD.append(distance)
            
            path.worldYaw.append(path.worldYaw[-1])
            path.worldD.append(path.worldD[-1])

            for i in range(len(path.worldYaw) - 1):
                if path.worldD[i] != 0:
                    path.maxCurvature.append((path.worldYaw[i + 1] - path.worldYaw[i]) / path.worldD[i])
            
        return frenetPaths


    def isColliding(self, path, obstacles):
        for i in range(len(obstacles[:, 0])):
            d = [((ix - obstacles[i, 0]) ** 2 + (iy - obstacles[i, 1]) ** 2) for (ix, iy) in zip(path.worldX, path.worldY)]
            collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False

        return True

    def checkValid(self, convertedPaths, obstacles):
        pass_idx = []
        for i, _ in enumerate(convertedPaths):
            if any([v > self.maxVelocity_ for v in convertedPaths[i].lonVel]):  # Max speed check
                continue
            elif any([abs(a) > self.maxAcceleration_ for a in convertedPaths[i].lonAcc]):  # Max accel check
                continue
            elif not self.isColliding(convertedPaths[i], obstacles):
                continue

            pass_idx.append(i)
        return [convertedPaths[i] for i in pass_idx]


    def optimalTrajectory(self, d0, dv0, da0, s0, sv0, sa0, globalPath, obstacles):
        trajectory = self.frenetTrajectory(d0, dv0, da0, s0, sv0, sa0)
        trajectory = self.globalCoordinates(trajectory, globalPath)
        trajectory = self.checkValid(trajectory, obstacles)

        bestPath = FrenetPath()
        minCost = float('inf')
        for path in trajectory:
            if minCost >= path.costFunction:
                minCost = path.costFunction
                bestPath = path

        return bestPath


    def run(self):
        self.generator.wayPoints()
        len_pathposes = len(self.generator.pathPoses)

        obstacles = np.array([[self.generator.pathPoses[int(len_pathposes * 0.2)][0], self.generator.pathPoses[int(len_pathposes * 0.2)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.25)][0], self.generator.pathPoses[int(len_pathposes * 0.25)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.5)][0], self.generator.pathPoses[int(len_pathposes * 0.5)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.6)][0], self.generator.pathPoses[int(len_pathposes * 0.6)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.8)][0], self.generator.pathPoses[int(len_pathposes * 0.8)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.82)][0], self.generator.pathPoses[int(len_pathposes * 0.82)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.84)][0], self.generator.pathPoses[int(len_pathposes * 0.84)][1]],
                    [self.generator.pathPoses[int(len_pathposes * 0.86)][0], self.generator.pathPoses[int(len_pathposes * 0.86)][1]],
                    ])   
        
        d0, dv0, da0, s0, sv0, sa0 = 0, 0, 0, 0, 0.0, 0.0  # Initial conditions
        plt.xlim([-10, 25])
        plt.ylim([-15, 15])
        while True:
            frenet_path = self.frenetTrajectory(d0, dv0, da0, s0, sv0, sa0)
            global_path = self.globalCoordinates(frenet_path, self.generator.pathPoses)

            # Get optimal Trajectory
            optimalPath = self.optimalTrajectory(d0, dv0, da0, s0, sv0, sa0, self.generator.pathPoses, obstacles)

            # Update current state to follow the path
            d0 = optimalPath.latPos[1]
            dv0 = optimalPath.latVel[1]
            da0 = optimalPath.latAcc[1]

            s0 = optimalPath.lonPos[1]
            sv0 = optimalPath.lonVel[1]
            sa0 = optimalPath.lonAcc[1]

            # Stop planner when within goal threshold
            goal_distance = np.sqrt((self.generator.pathPoses[-1][0] - optimalPath.worldX[1]) ** 2 + (self.generator.pathPoses[-1][1] - optimalPath.worldY[1]) ** 2)
            if goal_distance <= ROBOT_RADIUS:
                print("Goal Reached")
                break


            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            for fp in frenet_path:
                plt.plot(fp.lonPos, fp.latPos, "-r", alpha=0.1, label='Frenet Path')
            for fp in global_path:
                plt.plot(fp.worldX, fp.worldY, "-g", alpha=0.1, label='Checked Path')
            lane_x = [pt[0] for pt in self.generator.pathPoses]
            lane_y = [pt[1] for pt in self.generator.pathPoses]
            plt.plot(lane_x, lane_y, '-k')

            plt.plot(optimalPath.latPos[1], 'ob')

            plt.plot(obstacles[:, 0], obstacles[:, 1], "xk")

            # plt.plot(optimalPath.worldX[1:], optimalPath.worldY[1:], '-or')
            plt.plot(optimalPath.worldX[1], optimalPath.worldY[1], 'vc')

            plt.title("v[m/s]:" + str(sv0))
            plt.grid(True)
            plt.pause(0.0001)

        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    points = [(0, 0), (2, 4), (6, 5), (10, 3), (12, 4), (20, 0)]
    planner = OptimalTrajectoryPlanner(points)
    planner.run()