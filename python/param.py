
import numpy as np
import math 
import matplotlib.pyplot as plt
import os
#Ranger
RANGER_FOV=60/180*np.pi
RANGER_SAMPLES=8
RANGER_VALID_DISTANCE=[0.1,4]
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

        
# 替换原有的定义
MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX = [] 


#DATA_TEXT_FIGURE,DATA_TEXT_AX=plt.subplot()
#STATSTIC_FIGURE,STATISTIC_AX=plt.subplot()


#####################C param only##########################
C_MAP_POINTS=64
C_MAP_VALID_POINT_MIN_DIS=0.2
C_MAP_TARGETS=2
C_MAP_TARGETS_X=[8,0]#由于simulator尚不支持多目标，所以暂时不同步参数
C_MAP_TARGETS_Y=[8,0]

C_POS_VECTOR_LEN = 7
C_P0_DEFAULT = [1.0, 1.0, 0.2, 0.5, 0.5, 0.05, 0.01]
C_Q0_DEFAULT = [1e-6, 1e-6, 1e-7, 1e-3, 1e-3, 1e-6, 1e-8]

C_RANGER_FOV = RANGER_FOV
C_RANGER_SAMPLE = RANGER_SAMPLES
C_RANGER_MIN_DIST = RANGER_VALID_DISTANCE[0]
C_RANGER_MAX_DIST = RANGER_VALID_DISTANCE[1]

angle_step = (C_RANGER_FOV) / (C_RANGER_SAMPLE - 1)
C_RANGER_RAY_ANGLE_COS = math.cos(angle_step)
C_RANGER_RAY_ANGLE_SIN = math.sin(angle_step)

C_ISA_WORK_CYCLE = 0.01  # 100hz
C_DWA_WORK_CYCLE = 0.1   # 10hz

C_MAX_V=2
C_MAX_OMIGA=180/180*np.pi
C_MAX_A=1
C_MAX_ALPHA=2

C_DWA_DT=0.1
C_DWA_PREDICT_TIME=1.5
C_DWA_WEIGHT_GOAL=1
C_DWA_WEIGHT_OBSTACLE=1.5
C_DWA_WEIGHT_VELOCITY=0.3
C_DWA_MIN_DISTANCE_TO_OBSTACLE=DRONE_RADIUS
C_DWA_VELOCITY_SEARCH_STEP=0.1
C_DWA_ANGULAR_VELOCITY_STEP=0.1
C_PARAM_LIST = [
    ["MAP_POINTS", str(C_MAP_POINTS), ""],
    ["MAP_VALID_POINT_MIN_DIST", str(C_MAP_VALID_POINT_MIN_DIS), ""],
    ["MAP_TARGETS", str(C_MAP_TARGETS), ""],
    ["MAP_TARGETS_X", str(C_MAP_TARGETS_X).replace('[','{').replace(']','}'), ""],
    ["MAP_TARGETS_Y", str(C_MAP_TARGETS_Y).replace('[','{').replace(']','}'), ""],
    ["MAP_TARGET_RADIUS", "0.2", ""], # 示例中未给出 C_ 变量，可自行定义
    ["MAP_UPDATE_WORK_CYCLE","0.1","地图更新最小周期"],
    ["POS_VECTOR_LEN", str(C_POS_VECTOR_LEN), "[x y theta vx vy b_a前向0偏 b_g陀螺0偏]"],
    ["P0_DEFAULT", str(C_P0_DEFAULT).replace('[','{').replace(']','}'), "x y theta ax ay b_ax b_omiga"],
    ["Q0_DEFAULT", str(C_Q0_DEFAULT).replace('[','{').replace(']','}'), ""],
    ["RANGER_FOV", f"{C_RANGER_FOV }", ""],
    ["RANGER_SAMPLE", str(C_RANGER_SAMPLE), ""],
    ["RANGER_RAY_ANGLE_COS", f"{C_RANGER_RAY_ANGLE_COS:.11f}", "cos(fov/(sample-1))"],
    ["RANGER_RAY_ANGLE_SIN", f"{C_RANGER_RAY_ANGLE_SIN:.11f}", "sin(fov/(sample-1))"],
    ["RANGER_MIN_DIST", str(C_RANGER_MIN_DIST), ""],
    ["RANGER_MAX_DIST", str(C_RANGER_MAX_DIST), ""],
    ["INS_WORK_CYCLE", str(C_ISA_WORK_CYCLE), "100hz"],
    ["DWA_WORK_CYCLE", str(C_DWA_WORK_CYCLE), "10hz"],
    ["MAX_V",                 str(C_MAX_V),                 "最大速度 m/s"],
    ["MAX_OMIGA",             f"{C_MAX_OMIGA:.11f}",        "最大角速度 rad/s"],
    ["MAX_A",                 str(C_MAX_A),                 "最大加速度 m/s^2"],
    ["MAX_ALPHA",             str(C_MAX_ALPHA),             "最大角加速度 rad/s^2"],
    ["DWA_DT",                str(C_DWA_DT),                "DWA预测步长"],
    ["DWA_PREDICT_TIME",      str(C_DWA_PREDICT_TIME),      "DWA预测时长"],
    ["DWA_WEIGHT_GOAL",       str(C_DWA_WEIGHT_GOAL),       "目标点权重"],
    ["DWA_WEIGHT_OBSTACLE",   str(C_DWA_WEIGHT_OBSTACLE),   "障碍物权重"],
    ["DWA_WEIGHT_VELOCITY",   str(C_DWA_WEIGHT_VELOCITY),   "速度权重"],
    ["DWA_MIN_DISTANCE_TO_OBSTACLE",str(C_DWA_MIN_DISTANCE_TO_OBSTACLE),"距离障碍物最小距离"],
    ["DWA_VELOCITY_SEARCH_STEP",str(C_DWA_VELOCITY_SEARCH_STEP),"速度搜索步长"],
    ["DWA_ANGULAR_VELOCITY_SEARCH_STEP",str(C_DWA_ANGULAR_VELOCITY_STEP),"角速度搜索步长"]
    
]

HEAD_FILE_PATH=os.path.dirname(os.path.abspath(__file__))+"/../src/param.h"