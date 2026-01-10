import matplotlib.pyplot as plt
import numpy as np
from map import Map, Circle, Rectangle, Line, Ray
from param import *
from drone import Drone, Ranger



def main():
    """主函数：创建地图、无人机和激光雷达，并进行可视化"""
    # 创建地图
    map_instance = Map()
    map_instance.gen_map()
    
    # 创建无人机和激光雷达
    drone = Drone("test_drone", "sim",map=map_instance)
    
    # 设置激光雷达位置和方向
    x = np.random.uniform(-0.5,8.5)
    y = np.random.uniform(-0.5,8.5)
    theta = np.pi*np.random.uniform(0,8)/4  
    drone.set_pos(x,y,theta)
    plt.figure()
    map_instance.paint(color="black")
    drone.paint()
    plt.show()


if __name__ == "__main__":
    
    main()