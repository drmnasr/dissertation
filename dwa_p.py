import dwa
import numpy as np
import math
from numba import njit

@njit
def getObstacles(contours):
    obstacles = []

    for contour in contours:
        for point in contour:
            obstacles.append([point[0][0], point[0][1]])

    return obstacles

def navigate(pose, velocity, goal, contours, warped, angle):
    point_cloud = getObstacles(contours)

    config = dwa.Config(
        max_speed = 50,
        min_speed = -1.0,
        max_yawrate = np.radians(40.0),
        max_accel = 15.0,
        max_dyawrate = np.radians(110.0),
        velocity_resolution = 0.1,
        yawrate_resolution = np.radians(1.0),
        dt = 0.1,
        predict_time = 3.0,
        heading = 0.15,
        clearance = 1.0,
        velocity = 1.0,
        base = [-3.0, -2.5, +3.0, +2.5]
    )

    velocity = dwa.planning(pose, velocity, goal, np.array(point_cloud, np.float32), config)
    pose = dwa.motion(pose, velocity, config.dt)

    d =  80
    newX = pose[0] + int(d * math.cos(math.radians(angle)))
    newY = pose[1] + int(d * math.sin(math.radians(angle)))
    goal = (newX, newY)

    return pose, velocity, goal, angle

