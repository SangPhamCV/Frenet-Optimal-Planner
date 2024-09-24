import numpy as np
import matplotlib.pyplot as plt
from Polynomial import Polynomial


class CurveGenerator:
    def __init__(self, tension=0.5):
        self.points = []
        self.curves = []
        self.tension = tension
        self.path_poses = []
        self.ROBOT_RADIUS = 0.1

    def hermite_curve(self, p0, p1, t0, t1):
        distance = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        num_points = int(distance // self.ROBOT_RADIUS + 1)

        t = np.linspace(0, 1, num_points)
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
        y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]
        return x.tolist(), y.tolist()  # Ensure the return values are lists
            
    def create_tangent_points(self):
        self.tangent_points = []
        if len(self.points) < 2:
            return

        for i in range(len(self.points)):
            if i == 0:
                x_diff = self.points[1][0] - self.points[0][0]
                y_diff = self.points[1][1] - self.points[0][1]
            elif i == len(self.points) - 1:
                x_diff = self.points[i][0] - self.points[i - 1][0]
                y_diff = self.points[i][1] - self.points[i - 1][1]
            else:
                x_diff = self.points[i + 1][0] - self.points[i - 1][0]
                y_diff = self.points[i + 1][1] - self.points[i - 1][1]
            self.tangent_points.append(((1 - self.tension) * x_diff, (1 - self.tension) * y_diff))

    def create_curve(self):
        self.curves = []
        if len(self.points) < 2:
            return

        self.create_tangent_points()

        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            t0 = self.tangent_points[i]
            p1 = self.points[i + 1]
            t1 = self.tangent_points[i + 1]

            curve_x, curve_y = self.hermite_curve(p0, p1, t0, t1)
            self.curves.append((curve_x, curve_y))
        
    def way_points(self):
        self.create_curve()
        distance_traced = 0
        prev_x, prev_y = 0, 0
        self.path_poses = []  # Initialize the path poses
        for curve_idx, curve in enumerate(self.curves):
            curve_x, curve_y = curve
            for i in range(len(curve_x)):
                x, y = curve_x[i], curve_y[i]
                
                if i < len(curve_x) - 1:
                    x_next, y_next = curve_x[i+1], curve_y[i+1]
                else:
                    if curve_idx < len(self.curves) - 1:
                        next_curve = self.curves[curve_idx + 1]
                        x_next, y_next = next_curve[0][0], next_curve[1][0]
                    else:
                        break
                distance_traced += np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                yaw = np.arctan2(y_next - y, x_next - x)
                self.path_poses.append([x, y, yaw, distance_traced])
                prev_x, prev_y = x, y


class FrenetPath:
    def __init__(self):
        self.longitudinal = []       # Longitudinal Data
        self.latitudinal = []       # Latitudinal Data
        self.world = []
        self.predictionTime = 0.0      # Prediction Time
        self.costFunction = 0.0     # Cost
        self.jerkLongitudinal = 0.0
        self.jerkLatitudinal = 0.0
        self.maxVelocity = float('-inf')
        self.maxAcceleration = float('-inf')
        self.maxCurvature = float('-inf')


class OptimalTrajectoryPlanner():
    def __init__(self, points):
        self.generator = CurveGenerator()
        self.generator.points = points

        self.maxVelocity_ = 1.2
        self.maxAcceleration_ = 2
        self.maxCurvature_ = 3

        self.maxPredictionStep_ = 4
        self.minPredictionStep_ = 1.33
        
        self.noOfLanes_ = 5
        self.laneWidth_ = 0.66

        self.targetVelocity_ = 0.8
        self.velocityStep_ = 0.1
        self.timeStep_ = 0.4125

        self.klat_ = 1
        self.klon_ = 1
        self.kjd_ = 0.1
        self.ktd_ = 0.1
        self.ksd_ = 1
        self.kjs_ = 0.1
        self.kts_ = 0.1
        self.kss_ = 1

        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)
        self.ax.grid(True)

    def optimal_trajectory(self, d0, dv0, da0, s0, sv0,
                            center_lane, obstacles, all_paths):
        paths = []

        for T in np.arange(self.minPredictionStep_, self.maxPredictionStep_, 0.1):
            for dT in np.arange(-((self.noOfLanes_ - 1) * self.laneWidth_) / 2, 
                                ((self.noOfLanes_ - 1) * self.laneWidth_) / 2 + self.laneWidth_, 
                                self.laneWidth_):
                dvT, daT, jd = 0, 0, 0
                latitudinal_trajectory = []

                quintic = Polynomial(d0, dv0, da0, dT, dvT, daT, T)
                for t in np.arange(0, T + self.timeStep_, self.timeStep_):
                    data = [quintic.position(t), quintic.velocity(t), quintic.acceleration(t), quintic.jerk(t), t]
                    jd += np.power(data[3], 2)
                    latitudinal_trajectory.append(data)

                for svT in np.arange(self.targetVelocity_ - self.velocityStep_, 
                                    self.targetVelocity_ + self.velocityStep_ + self.velocityStep_, 
                                    self.velocityStep_):
                    path = FrenetPath()
                    path.predictionTime = T
                    path.latitudinal = latitudinal_trajectory
                    path.jerkLatitudinal = jd
                    longitudinal_trajectory = []
                    quartic = Polynomial(s0, sv0, 0, xT=None, vT=svT, aT=0, T=T)  # Ensure T is not None
                    js = 0
                    for t in np.arange(0, T + self.timeStep_, self.timeStep_):
                        data = [quartic.position(t), quartic.velocity(t), quartic.acceleration(t), quartic.jerk(t), t]
                        js += np.power(data[3], 2)
                        if data[1] > path.maxVelocity:
                            path.maxVelocity = data[1]
                        if data[2] > path.maxAcceleration:
                            path.maxAcceleration = data[2]
                        longitudinal_trajectory.append(data)
                    path.longitudinal = longitudinal_trajectory
                    path.jerkLongitudinal = js

                    cd = path.jerkLatitudinal * self.kjd_ + path.predictionTime * self.ktd_ + np.power(path.latitudinal[-1][0], 2) * self.ksd_
                    cs = path.jerkLongitudinal * self.kjs_ + path.predictionTime * self.kts_ + np.power(path.longitudinal[0][0] - path.longitudinal[-1][0], 2) * self.kss_

                    path.costFunction = self.klat_ * cd + self.klon_ * cs
    
                    paths.append(path)

        all_paths.extend(paths)
        self.convert_to_world_frame(paths, center_lane)
        valid_paths = self.is_valid(paths, obstacles)

        optimal_trajectory = FrenetPath()
        cost = float('inf')
        for path in valid_paths:
            if cost >= path.costFunction:
                cost = path.costFunction
                optimal_trajectory = path
        
        return optimal_trajectory
    
    def convert_to_world_frame(self, paths, center_lane):
        for path in paths:
            j = 0
            for i in range(len(path.longitudinal)):
                x, y, yaw = 0, 0, 0
                while j < len(center_lane) and abs(path.longitudinal[i][0] - center_lane[j][3]) > 0.1:
                    j += 1
                if j < len(center_lane):
                    x = center_lane[j][0]
                    y = center_lane[j][1]
                    yaw = center_lane[j][2]
                d = path.latitudinal[i][0]
                path.world.append([x + d * np.cos(yaw + 1.57), y + d * np.sin(yaw + 1.57), 0, 0])

            # Ensure that distances between consecutive points are non-zero to avoid division by zero
            min_distance = 1e-6  # Small threshold to avoid division by zero
            for i in range(len(path.world) - 1):
                dx = path.world[i + 1][0] - path.world[i][0]
                dy = path.world[i + 1][1] - path.world[i][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < min_distance:
                    distance = min_distance  # Set to a small non-zero value to prevent division by zero

                path.world[i][2] = np.arctan2(dy, dx)
                path.world[i][3] = distance
            
            path.world[-1][2] = path.world[-2][2]
            path.world[-1][3] = path.world[-2][3]

            # Calculate curvature with a safe check for division by zero
            curvature = float('-inf')
            for i in range(len(path.world) - 1):
                dx = path.world[i + 1][0] - path.world[i][0]
                dy = path.world[i + 1][1] - path.world[i][1]
                distance = path.world[i][3]
                if distance < min_distance:
                    distance = min_distance  # Set to a small non-zero value to prevent division by zero

                temp_curvature = abs((path.world[i + 1][2] - path.world[i][2]) / distance)
                if curvature < temp_curvature:
                    curvature = temp_curvature
            path.maxCurvature = curvature


    def is_colliding(self, path, obstacles):
        for point in path.world:
            for obs in obstacles:
                if np.sqrt(np.power(point[0] - obs[0], 2) + np.power(point[1] - obs[1], 2)) <= 1:
                    return True
        return False

    def is_within_kinematic_constraints(self, path):
        return path.maxVelocity <= self.maxVelocity_ and \
               path.maxAcceleration <= self.maxAcceleration_ and \
               path.maxCurvature <= self.maxCurvature_

    def is_valid(self, paths, obstacles):
        valid_paths = []
        for path in paths:
            if self.is_within_kinematic_constraints(path):
            # if not self.is_colliding(path, obstacles) and self.is_within_kinematic_constraints(path):
                valid_paths.append(path)
        return valid_paths

    def run(self):
        self.generator.way_points()

        obstacles = [
                # [plot.path_poses[200][0], plot.path_poses[200][1]],
                # [self.generator.path_poses[150][0], self.generator.path_poses[150][1]],
                # [self.generator.path_poses[601][0], self.generator.path_poses[601][1]],
                # [self.generator.path_poses[900][0], self.generator.path_poses[900][1]],

                # [self.generator.path_poses[1200][0], self.generator.path_poses[1200][1]],
                [0, 0]
                ]
        
        d0, dv0, da0, s0, sv0 = 0, 0, 0, 0, 0  # Initial conditions

        while True:
            # inst  = input("Press 'y' to clear obstacles: ")
            # if inst == 'y':
            #     obstacles = [
            #             # [plot.path_poses[200][0], plot.path_poses[200][1]],
            #             [self.generator.path_poses[150][0], self.generator.path_poses[150][1]],
            #             [self.generator.path_poses[900][0], self.generator.path_poses[900][1]],

            #             [self.generator.path_poses[1200][0], self.generator.path_poses[1200][1]],
            #             ]
            #     # Get optimal Trajectory
            #     all_paths = []  # You need to replace this with actual trajectory retrieval
            #     p = self.optimal_trajectory(d0, dv0, da0, s0, sv0, self.generator.path_poses, obstacles, all_paths)

            #     d0 = p.latitudinal[1][0]
            #     dv0 = p.latitudinal[1][1]
            #     da0 = p.latitudinal[1][2]
            #     s0 = p.longitudinal[1][0]
            #     sv0 = p.longitudinal[1][1]
            #     x = p.world[1][0]
            #     y = p.world[1][1]

            #     # Stop planner when within goal threshold
            #     if np.sqrt((self.generator.path_poses[-1][0] - x) ** 2 + (self.generator.path_poses[-1][1] - y) ** 2) <= 1:
            #         break

            #     self.update_plot(obstacles, p.world, all_paths)                
            # else:
                # Get optimal Trajectory
                all_paths = []  # You need to replace this with actual trajectory retrieval
                p = self.optimal_trajectory(d0, dv0, da0, s0, sv0, self.generator.path_poses, obstacles, all_paths)

                d0 = p.latitudinal[1][0]
                dv0 = p.latitudinal[1][1]
                da0 = p.latitudinal[1][2]
                s0 = p.longitudinal[1][0]
                sv0 = p.longitudinal[1][1]

                # Stop planner when within goal threshold
                if np.sqrt((self.generator.path_poses[-1][0] - p.world[1][0]) ** 2 + (self.generator.path_poses[-1][1] - p.world[1][1]) ** 2) <= 0.5:
                    break

                self.update_plot(obstacles, p.world, all_paths)

    def update_plot(self, obstacles, p_world, all_paths):
        self.ax.clear()
                
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        # Lane
        lane_x = [pt[0] for pt in self.generator.path_poses]
        lane_y = [pt[1] for pt in self.generator.path_poses]
        self.ax.plot(lane_x, lane_y, color='black', linewidth=2)

        # Obstacles
        for obs in obstacles:
            self.ax.plot(obs[0], obs[1], 'ro', markersize=10)

        # Robot position
        self.ax.plot(p_world[0][0], p_world[0][1], 'bo', markersize=6)
        
        # All Trajectories
        for t in all_paths:

            traj_x = [pt[0] for pt in t.world]
            traj_y = [pt[1] for pt in t.world]

            frenet_x = [pt[0] for pt in t.longitudinal]
            frenet_y = [pt[0] for pt in t.latitudinal]

            self.ax.plot(traj_x, traj_y, 'b-', linewidth=1)
            self.ax.plot(frenet_x, frenet_y, 'r-', linewidth=1)

        # Trajectory
        path_x = [pt[0] for pt in p_world]
        path_y = [pt[1] for pt in p_world]
        self.ax.plot(path_x, path_y, 'g-', linewidth=2)

        plt.pause(0.001)  # Use pause instead of show for continuous update

if __name__ == "__main__":
    points = [(-3, -3), (2, 4), (6, 5), (10, 3), (12, 4), (20, 0)]
    planner = OptimalTrajectoryPlanner(points)
    planner.run()


