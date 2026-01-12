import matplotlib.pyplot as plt
import numpy as np
from map import Map, Circle, Rectangle, Line, Ray
from param import *
from drone import Drone, Ranger



def main():
    map_instance = Map()
    r=Ranger(map=map_instance)
    drone = Drone("test_drone", ranger=r)
    
    # 设置激光雷达位置和方向
    x = 3#np.random.uniform(-0.5,8.5)
    y = 3#np.random.uniform(-0.5,8.5)
    theta = 0#np.pi*np.random.uniform(0,8)/4  
    drone.set_pos(x,y,theta)
    plt.figure()
    map_instance.paint(color="black")
    drone.paint()
    plt.show()


if __name__ == "__main__":
    
    main()