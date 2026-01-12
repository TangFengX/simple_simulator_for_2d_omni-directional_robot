
import numpy as np
import math 
import matplotlib.pyplot as plt
import os
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
DRONE_X_INITIAL,DRONE_Y_INITIAL=(0,0)
DRONE_THETA_INITIAL=0
DRONE_LA_NOISE_VAR=0.01
DRONE_HA_NOISE_VAR=0.01
DRONE_IMU_OMIGA_NOISE_VAR=0.01
#Simulator
SIMULATOR_DT=0.01
SIMULATOR_LOG_PATH=os.path.dirname(os.path.abspath(__file__))+"/../log/"
SIMULATOR_LOG_OUTPUT_TERMINAL=True
SIMULATOR_MAX_TIME=-1 

#Monitor
MONITOR_MAP_FIGURE,MONITOR_MAP_AX=plt.subplots()
MONITOR_MAP_GLOBAL_ALL_ARTISTS=[]
RATIO_OF_PHYSCAL_FRAME_TO_MONITER_FRAME=30
import traceback

class TraceableList(list):
    def append(self, item):
        super().append(item)

    def clear(self):
        # 打印是谁调用了清理
        print("\n" + "="*30)
        print("DETECTED CLEAR() OPERATION!")
        traceback.print_stack() # 打印函数调用栈
        print("="*30 + "\n")
        super().clear()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        
# 替换原有的定义
# MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX = [] 
MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX = TraceableList()

#DATA_TEXT_FIGURE,DATA_TEXT_AX=plt.subplot()
#STATSTIC_FIGURE,STATISTIC_AX=plt.subplot()
