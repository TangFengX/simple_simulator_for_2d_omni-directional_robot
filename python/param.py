
import numpy as np
import math 
#Ranger
RANGER_FOV=60/180*np.pi
RANGER_SAMPLES=8
RANGER_VALID_DISTANCE=[0,4]
RANGER_INVALID_DISTANCE_MARK=-1


#Map
MAP_X_RANGE=[-1,9]
MAP_Y_RANGE=[-1,9]
MAP_WITH_BOUNDARY=True
MAP_ENTITIES_CONFIG = [
    {"type": "Circle", "center": [2, 2], "radius": 0.5},
    {"type": "Circle", "center": [6, 7], "radius": 0.7},
    {"type": "Circle", "center": [4, 4], "radius": 0.6},
    {"type": "Rectangle", "x": 1, "y": 7, "width": 1, "height": 0.5, "theta": 0},
    {"type": "Rectangle", "x": 7, "y": 1, "width": 1.2, "height": 0.6, "theta": math.pi/6},
    {"type": "Rectangle", "x": 5, "y": 5, "width": 0.8, "height": 0.8, "theta": math.pi/4},
    {"type": "Circle", "center": [7, 4], "radius": 0.5},
    {"type": "Circle", "center": [3, 6], "radius": 0.4},
    {"type": "Rectangle", "x": 2, "y": 5, "width": 0.6, "height": 1, "theta": math.pi/8},
    {"type": "Rectangle", "x": 6, "y": 3, "width": 0.5, "height": 1.2, "theta": -math.pi/6},
]
MAP_TARGET_POSITION=[8,8]
MAP_TARGET_RADIUS=0.2
#Print
PRINT_RAY_LENGTH_DEFAULT=10
#Drone
DRONE_RADIUS=0.1
DRONE_PAINT_RANGER_RAYS=True