import ctypes
import numpy as np
from param import *
class Pilot:
    def __init__(self):
        self.target=MAP_TARGET_POSITION
    def get_order(self,imu:list[float],ranger:list[float],time:float):
        al=0.1
        ah=0
        alpha=0.05
        return al,ah,alpha