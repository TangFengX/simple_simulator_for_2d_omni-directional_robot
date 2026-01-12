import numpy as np
from typing import List, Union
import math
import matplotlib.pyplot as plt
from param import *
class Circle:
    def __init__(self,center:list[float,float]=[0,0],radius:float=1,**kwargs):
      self.center=center
      self.radius=radius
      self.artist_no=len(MONITOR_MAP_GLOBAL_ALL_ARTISTS)
      graph,=plt.plot([],[],color="black")
      MONITOR_MAP_GLOBAL_ALL_ARTISTS.append(graph)
    def set_pos(self,center:list[float,float]=None,radius:float=None):
      if center is not None:
        self.center=center 
      if radius is not None:
        self.radius=radius
    def paint(self, **kwargs):
        """绘制圆"""
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        plt.plot(x, y, **kwargs)
    def draw(self,color=None,linewidth=None,linestyle=None,alpha=None,active=False):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_data(x,y)
        if color is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_color(color)
        if linewidth is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linewidth(linewidth)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linestyle(linestyle)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_alpha(alpha)
        if active:
            MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.append(self.artist_no)



class Rectangle:
    def __init__(self, x: float=0, y: float=0, width: float=1, height: float=1, theta: float = 0.0):
        """
        构造一个矩形
        :param x: 中心 x 坐标
        :param y: 中心 y 坐标
        :param width: 矩形宽度
        :param height: 矩形高度
        :param theta: 旋转角（弧度，逆时针为正）
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theta = theta
        self.artist_no=len(MONITOR_MAP_GLOBAL_ALL_ARTISTS)
        graph,=plt.plot([],[],color="black")
        MONITOR_MAP_GLOBAL_ALL_ARTISTS.append(graph)


    def set_pos(self, x: float=None, y: float=None, width: float=None, height: float=None, theta: float=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if theta is not None:
            self.theta = theta  

    def get_corners(self):
        """
        获取矩形四角的世界坐标，按顺时针顺序返回
        :return: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        hw = self.width / 2
        hh = self.height / 2
        local_corners = [
            (-hw, -hh),  # 左下
            ( hw, -hh),  # 右下
            ( hw,  hh),  # 右上
            (-hw,  hh)   # 左上
        ]

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        world_corners = []
        for lx, ly in local_corners:
            wx = self.x + lx * cos_t - ly * sin_t
            wy = self.y + lx * sin_t + ly * cos_t
            world_corners.append((wx, wy))

        return world_corners
    def paint(self, **kwargs):
        corners = self.get_corners() + [self.get_corners()[0]]  # 封闭矩形
        x, y = zip(*corners)
        plt.plot(x, y, **kwargs)
        
    def draw(self,color=None,linewidth=None,linestyle=None,alpha=None,active=False):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        corners = self.get_corners() + [self.get_corners()[0]]  # 封闭矩形
        x, y = zip(*corners)
        MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_data(x,y)
        if color is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_color(color)
        if linewidth is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linewidth(linewidth)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linestyle(linestyle)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_alpha(alpha)
        if active:
            MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.append(self.artist_no)
    
        
    

    

class Line:
    def __init__(self,a:float=1,b:float=0,inf_k:bool=False):
        """
        y=ax+b 
        or
        x=a when inf_k=True
        """
        self.a=a
        self.b=b
        self.inf_k=inf_k
        self.is_valid=True
        self.artist_no=len(MONITOR_MAP_GLOBAL_ALL_ARTISTS)
        graph,=plt.plot([],[],color="black")
        MONITOR_MAP_GLOBAL_ALL_ARTISTS.append(graph)
        
    def set_pos(self,a:float=None,b:float=None,inf_k:bool=None):
        if a is not None:
            self.a=a
        if b is not None:
            self.b=b
        if inf_k is not None:
            self.inf_k=inf_k
    def set_with_point(self, p1: list[float], p2: list[float]):
        """
        根据两点确定直线方程
        :param p1: [x1, y1]
        :param p2: [x2, y2]
        """
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2 and y1 == y2:
            self.is_valid = False
            self.inf_k = False
            self.a = 0
            self.b = 0
            print("Warning: two points are the same, line is invalid.")
            return

        self.is_valid = True
        if x1 == x2:

            self.inf_k = True
            self.a = x1   
            self.b = 0    
        else:
            self.inf_k = False
            self.a = (y2 - y1) / (x2 - x1)   
            self.b = y1 - self.a * x1 
        
    def paint(self, xlim=None, ylim=None, **kwargs):
        """在已有 plt 上绘制直线"""
        if not self.is_valid:
            return
        if xlim is None:
            xlim = plt.gca().get_xlim()
        if ylim is None:
            ylim = plt.gca().get_ylim()

        if self.inf_k:
            x = [self.a, self.a]
            y = ylim
        else:
            x = np.array(xlim)
            y = self.a * x + self.b
        plt.plot(x, y, **kwargs)
    def draw(self, xlim=None, ylim=None,color=None,linewidth=None,linestyle=None,alpha=None,active=False):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        if not self.is_valid:
            return
        if xlim is None:
            xlim = plt.gca().get_xlim()
        if ylim is None:
            ylim = plt.gca().get_ylim()

        if self.inf_k:
            x = [self.a, self.a]
            y = ylim
        else:
            x = np.array(xlim)
            y = self.a * x + self.b
        MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_data(x,y)
        if color is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_color(color)
        if linewidth is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linewidth(linewidth)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linestyle(linestyle)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_alpha(alpha)
        if active:
            MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.append(self.artist_no)

class Ray:
    def __init__(self, origin: list[float,float], angle: float):
        """
        :param origin: 射线起点 [x0, y0]
        :param angle: 射线方向角（弧度，0指向x正方向，逆时针为正）
        """
        self.origin = origin
        self.angle = angle
        self.dir = [math.cos(angle), math.sin(angle)]  # 单位方向向量
        self.artist_no=len(MONITOR_MAP_GLOBAL_ALL_ARTISTS)
        graph,=plt.plot([],[],color="black")
        MONITOR_MAP_GLOBAL_ALL_ARTISTS.append(graph)

    def point_at(self, t: float) -> list[float,float]:
        """射线上 t 位置的点"""
        x0, y0 = self.origin
        dx, dy = self.dir
        return [x0 + t * dx, y0 + t * dy]
    
    def set_pos(self,origin: list[float,float], angle: float):
        self.origin = origin
        self.angle = angle
        self.dir = [math.cos(angle), math.sin(angle)]
    
    def paint(self, length=PRINT_RAY_LENGTH_DEFAULT, **kwargs):
        """绘制射线，默认长度10"""
        end = self.point_at(length)
        plt.plot([self.origin[0], end[0]], [self.origin[1], end[1]], **kwargs)
        
    def draw(self, length=PRINT_RAY_LENGTH_DEFAULT,color=None,linewidth=None,linestyle=None,alpha=None,active=False):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        end = self.point_at(length)
        x=[self.origin[0], end[0]]
        y=[self.origin[1], end[1]]
        MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_data(x,y)
        if color is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_color(color)
        if linewidth is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linewidth(linewidth)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_linestyle(linestyle)
        if linestyle is not None:
            MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no].set_alpha(alpha)
        if active:
            MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.append(self.artist_no)
# -------------------------------
# Line × Line
# -------------------------------
def line_line_intersection(l1, l2) -> List[list[float,float]]:
    if getattr(l1, 'is_valid', True) is False or getattr(l2, 'is_valid', True) is False:
        return []

    # 两条垂直线
    if l1.inf_k and l2.inf_k:
        if l1.a == l2.a:
            return []  # 重合不算交点
        else:
            return []  # 平行
    # l1垂直
    if l1.inf_k:
        x = l1.a
        y = l2.a * x + l2.b if not l2.inf_k else None
        return [[x, y]]
    # l2垂直
    if l2.inf_k:
        x = l2.a
        y = l1.a * x + l1.b
        return [[x, y]]
    # 两条普通直线
    if l1.a == l2.a:
        return []  # 平行
    x = (l2.b - l1.b) / (l1.a - l2.a)
    y = l1.a * x + l1.b
    return [[x, y]]


# -------------------------------
# Line × Segment
# -------------------------------
def line_segment_intersection(p1: list[float,float], p2: list[float,float], line) -> List[list[float,float]]:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return []  # 重合点线段无交点

    if line.inf_k:
        x = line.a
        if dx == 0:
            return []  # 平行或重合
        t = (x - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            return [[x, y]]
        return []
    else:
        a, b = line.a, line.b
        if dx == 0:
            x = x1
            y = a * x + b
            if min(y1, y2) <= y <= max(y1, y2):
                return [[x, y]]
            return []
        if dy == 0:
            y = y1
            if a == 0:
                return []  # 重合不算交点
            x = (y - b) / a
            if min(x1, x2) <= x <= max(x1, x2):
                return [[x, y]]
            return []
        # 一般情况，求 t
        t = ((a*x1 + b - y1) / (dy - a*dx)) if (dy - a*dx)!=0 else None
        if t is not None and 0 <= t <= 1:
            x = x1 + t * dx
            y = y1 + t * dy
            return [[x, y]]
        return []


# -------------------------------
# Line × Rectangle
# -------------------------------
def rectangle_line_intersection(rect, line) -> List[list[float,float]]:
    corners = rect.get_corners()
    points = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        pts = line_segment_intersection(p1, p2, line)
        points.extend(pts)
    # 去重（numpy比较）
    if points:
        points = np.unique(np.array(points), axis=0).tolist()
    return points


# -------------------------------
# Circle × Line
# -------------------------------
def circle_line_intersection(circle, line) -> List[list[float,float]]:
    cx, cy = circle.center
    r = circle.radius
    points = []

    if line.inf_k:
        x = line.a
        delta = r**2 - (x - cx)**2
        if delta < 0:
            return []
        y1 = cy + np.sqrt(delta)
        y2 = cy - np.sqrt(delta)
        if delta == 0:
            points.append([x, y1])
        else:
            points.append([x, y1])
            points.append([x, y2])
    else:
        a, b = line.a, line.b
        A = 1 + a**2
        B = 2*(a*(b - cy) - cx)
        C = cx**2 + (b - cy)**2 - r**2
        delta = B**2 - 4*A*C
        if delta < 0:
            return []
        elif delta == 0:
            x = -B/(2*A)
            y = a*x + b
            points.append([x, y])
        else:
            sqrt_delta = np.sqrt(delta)
            x1 = (-B + sqrt_delta)/(2*A)
            y1 = a*x1 + b
            x2 = (-B - sqrt_delta)/(2*A)
            y2 = a*x2 + b
            points.append([x1, y1])
            points.append([x2, y2])
    return points


# -------------------------------
# Rectangle × Rectangle
# -------------------------------
def rectangle_rectangle_intersection(r1, r2) -> List[list[float,float]]:
    points = []
    c1 = r1.get_corners()
    c2 = r2.get_corners()
    # r1的每条边和r2的每条边求交
    for i in range(4):
        p1, p2 = c1[i], c1[(i+1)%4]
        for j in range(4):
            q1, q2 = c2[j], c2[(j+1)%4]
            # 用线段交点函数
            temp_line = Line()
            temp_line.set_with_point(p1, p2)
            pts = line_segment_intersection(q1, q2, temp_line)
            points.extend(pts)
    if points:
        points = np.unique(np.array(points), axis=0).tolist()
    return points


# -------------------------------
# Circle × Rectangle
# -------------------------------
def circle_rectangle_intersection(circle, rect) -> list[list[float, float]]:
    points = []
    corners = rect.get_corners()
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        
        # 1. 创建描述该边的直线
        temp_line = Line()
        temp_line.set_with_point(p1, p2)
        
        line_intersects = circle_line_intersection(circle, temp_line)
        

        for pt in line_intersects:
            x, y = pt

            if (min(p1[0], p2[0]) - 1e-9 <= x <= max(p1[0], p2[0]) + 1e-9 and
                min(p1[1], p2[1]) - 1e-9 <= y <= max(p1[1], p2[1]) + 1e-9):
                points.append(pt)
                
    # 去重
    if points:

        points = [list(x) for x in set(tuple(np.round(p, 9)) for p in points)]
    return points


# -------------------------------
# Circle × Circle
# -------------------------------
def circle_circle_intersection(c1, c2) -> List[list[float,float]]:
    x0, y0 = c1.center
    r0 = c1.radius
    x1, y1 = c2.center
    r1 = c2.radius

    dx = x1 - x0
    dy = y1 - y0
    d = np.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return []  # 不相交或同心

    # 两圆交点公式
    a = (r0**2 - r1**2 + d**2) / (2*d)
    h = np.sqrt(r0**2 - a**2)
    xm = x0 + a*dx/d
    ym = y0 + a*dy/d
    xs1 = xm + h*dy/d
    ys1 = ym - h*dx/d
    xs2 = xm - h*dy/d
    ys2 = ym + h*dx/d
    if np.isclose(xs1, xs2) and np.isclose(ys1, ys2):
        return [[xs1, ys1]]
    else:
        return [[xs1, ys1], [xs2, ys2]]

def ray_line_intersection(ray: Ray, line) -> List[list[float,float]]:
    x0, y0 = ray.origin
    dx, dy = ray.dir
    if line.inf_k:
        denom = dx
        if np.isclose(denom, 0):
            return []
        t = (line.a - x0) / denom
    else:
        denom = dy - line.a * dx
        if np.isclose(denom, 0):
            return []
        t = (line.a * x0 + line.b - y0) / denom
    if t < 0:
        return []
    return [ray.point_at(t)]

def ray_circle_intersection(ray: Ray, circle) -> List[list[float,float]]:
    x0, y0 = ray.origin
    dx, dy = ray.dir
    cx, cy = circle.center
    r = circle.radius

    D = (x0 - cx)*dx + (y0 - cy)*dy
    E = (x0 - cx)**2 + (y0 - cy)**2 - r**2
    delta = D**2 - E
    if delta < 0:
        return []
    sqrt_delta = math.sqrt(delta)
    t1 = -D + sqrt_delta
    t2 = -D - sqrt_delta
    points = []
    for t in [t1, t2]:
        if t >= 0:
            points.append(ray.point_at(t))
    return points

def ray_segment_intersection(ray: Ray, p1: list[float,float], p2: list[float,float]) -> List[list[float,float]]:
    """
    计算射线与线段的交点
    Args:
        ray: 射线对象
        p1: 线段起点 [x1, y1]
        p2: 线段终点 [x2, y2]
    Returns:
        交点列表 [[x, y]]
    """
    x0, y0 = ray.origin
    dx, dy = ray.dir
    x1, y1 = p1
    x2, y2 = p2

    # 线段向量
    seg_dx = x2 - x1
    seg_dy = y2 - y1

    # 解方程组:
    # x0 + t * dx = x1 + s * seg_dx
    # y0 + t * dy = y1 + s * seg_dy
    det = dx * (-seg_dy) - dy * (-seg_dx)
    
    if np.isclose(det, 0):
        return []  # 平行或重合
    
    # 计算参数t和s
    t = ((x1 - x0) * (-seg_dy) - (y1 - y0) * (-seg_dx)) / det
    s = (dx * (y1 - y0) - dy * (x1 - x0)) / det
    
    # 检查是否在射线上(t >= 0)和线段上(0 <= s <= 1)
    if t >= 0 and 0 <= s <= 1:
        return [ray.point_at(t)]
    return []

def ray_rectangle_intersection(ray: Ray, rect) -> List[list[float,float]]:
    points = []
    corners = rect.get_corners()
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        pts = ray_segment_intersection(ray, p1, p2)
        points.extend(pts)
    # 去重
    if points:
        points = np.unique(np.array(points), axis=0).tolist()
    return points


# ===============================
# Collision Detection (快速碰撞检测，只判断是否相交，不计算具体交点)
# ===============================
def is_colliding(obj1: Union[Rectangle, Circle, Line, Ray],
                 obj2: Union[Rectangle, Circle, Line, Ray]) -> bool:
    """
    判断两种几何对象是否发生碰撞（只判断是否有交点，不计算具体交点）
    
    Args:
        obj1: 第一个几何对象
        obj2: 第二个几何对象
        
    Returns:
        bool: 是否发生碰撞
    """
    
    # Ray 相关碰撞检测
    if isinstance(obj1, Ray) and isinstance(obj2, Line):
        return _ray_line_collide(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Line):
        return _ray_line_collide(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Circle):
        return _ray_circle_collide(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Circle):
        return _ray_circle_collide(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Rectangle):
        return _ray_rectangle_collide(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Rectangle):
        return _ray_rectangle_collide(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Ray):
        return _ray_ray_collide(obj1, obj2)
    
    # Line 相关碰撞检测
    if isinstance(obj1, Line) and isinstance(obj2, Line):
        return _line_line_collide(obj1, obj2)
    elif isinstance(obj1, Line) and isinstance(obj2, Rectangle):
        return _rectangle_line_collide(obj2, obj1)
    elif isinstance(obj2, Line) and isinstance(obj1, Rectangle):
        return _rectangle_line_collide(obj1, obj2)
    elif isinstance(obj1, Line) and isinstance(obj2, Circle):
        return _circle_line_collide(obj2, obj1)
    elif isinstance(obj2, Line) and isinstance(obj1, Circle):
        return _circle_line_collide(obj1, obj2)
    
    # Rectangle-Rectangle 碰撞检测
    elif isinstance(obj1, Rectangle) and isinstance(obj2, Rectangle):
        return _rectangle_rectangle_collide(obj1, obj2)
    
    # Circle-Rectangle 碰撞检测
    elif isinstance(obj1, Circle) and isinstance(obj2, Rectangle):
        return _circle_rectangle_collide(obj1, obj2)
    elif isinstance(obj1, Rectangle) and isinstance(obj2, Circle):
        return _circle_rectangle_collide(obj2, obj1)
    
    # Circle-Circle 碰撞检测
    elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
        return _circle_circle_collide(obj1, obj2)
    
    print(f"Collision detection not implemented for this type combination: {type(obj1)} and {type(obj2)}")
    return False

def _ray_line_collide(ray: Ray, line) -> bool:
    """射线与直线碰撞检测"""
    x0, y0 = ray.origin
    dx, dy = ray.dir
    
    if line.inf_k:
        denom = dx
        if np.isclose(denom, 0):
            return False
        t = (line.a - x0) / denom
        return t >= 0
    else:
        denom = dy - line.a * dx
        if np.isclose(denom, 0):
            return False
        t = (line.a * x0 + line.b - y0) / denom
        return t >= 0

def _ray_circle_collide(ray: Ray, circle) -> bool:
    """射线与圆碰撞检测"""
    x0, y0 = ray.origin
    dx, dy = ray.dir
    cx, cy = circle.center
    r = circle.radius

    D = (x0 - cx) * dx + (y0 - cy) * dy
    E = (x0 - cx)**2 + (y0 - cy)**2 - r**2
    delta = D**2 - E
    
    if delta < 0:
        return False
    
    sqrt_delta = math.sqrt(delta)
    t1 = -D + sqrt_delta
    t2 = -D - sqrt_delta
    
    # 检查是否有正的t值
    return t1 >= 0 or t2 >= 0

def _ray_rectangle_collide(ray: Ray, rect) -> bool:
    """射线与矩形碰撞检测"""
    corners = rect.get_corners()
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        if _ray_segment_collide(ray, p1, p2):
            return True
    return False

def _ray_ray_collide(ray1: Ray, ray2: Ray) -> bool:
    """射线与射线碰撞检测"""
    x1, y1 = ray1.origin
    dx1, dy1 = ray1.dir
    x2, y2 = ray2.origin
    dx2, dy2 = ray2.dir

    # 解方程组:
    # x1 + t1 * dx1 = x2 + t2 * dx2
    # y1 + t1 * dy1 = y2 + t2 * dy2
    det = dx1 * (-dx2) - dy1 * (-dy2)
    
    if np.isclose(det, 0):
        return False  # 平行或重合
    
    # 计算参数t1和t2
    t1 = ((x2 - x1) * (-dx2) - (y2 - y1) * (-dy2)) / det
    t2 = (dx1 * (y2 - y1) - dy1 * (x2 - x1)) / det
    
    # 检查是否都在射线上(t >= 0)
    return t1 >= 0 and t2 >= 0

def _ray_segment_collide(ray: Ray, p1: list, p2: list) -> bool:
    """射线与线段碰撞检测"""
    x0, y0 = ray.origin
    dx, dy = ray.dir
    x1, y1 = p1
    x2, y2 = p2

    seg_dx = x2 - x1
    seg_dy = y2 - y1

    det = dx * (-seg_dy) - dy * (-seg_dx)
    
    if np.isclose(det, 0):
        return False
    
    t = ((x1 - x0) * (-seg_dy) - (y1 - y0) * (-seg_dx)) / det
    s = (dx * (y1 - y0) - dy * (x1 - x0)) / det
    
    return t >= 0 and 0 <= s <= 1

def _line_line_collide(l1, l2) -> bool:
    """直线与直线碰撞检测"""
    if getattr(l1, 'is_valid', True) is False or getattr(l2, 'is_valid', True) is False:
        return False

    # 两条垂直线
    if l1.inf_k and l2.inf_k:
        return l1.a == l2.a  # 重合也算碰撞
    # l1垂直
    if l1.inf_k:
        return True
    # l2垂直
    if l2.inf_k:
        return True
    # 两条普通直线
    return l1.a == l2.a  # 平行也算碰撞

def _rectangle_line_collide(rect, line) -> bool:
    """矩形与直线碰撞检测"""
    corners = rect.get_corners()
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        if _segment_line_collide(p1, p2, line):
            return True
    return False

def _segment_line_collide(p1: list, p2: list, line) -> bool:
    """线段与直线碰撞检测"""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return False

    if line.inf_k:
        x = line.a
        if dx == 0:
            return x == x1  # 重合
        t = (x - x1) / dx
        return 0 <= t <= 1
    else:
        a, b = line.a, line.b
        if dx == 0:
            x = x1
            y = a * x + b
            return min(y1, y2) <= y <= max(y1, y2)
        if dy == 0:
            y = y1
            if a == 0:
                return y == b  # 重合
            x = (y - b) / a
            return min(x1, x2) <= x <= max(x1, x2)
        # 一般情况，求 t
        denom = dy - a * dx
        if np.isclose(denom, 0):
            # 平行，检查是否重合
            y_on_line = a * x1 + b
            return np.isclose(y1, y_on_line)
        t = (a * x1 + b - y1) / denom
        return 0 <= t <= 1

def _circle_line_collide(circle, line) -> bool:
    """圆与直线碰撞检测"""
    cx, cy = circle.center
    r = circle.radius

    if line.inf_k:
        x = line.a
        distance = abs(x - cx)
        return distance <= r
    else:
        a, b = line.a, line.b
        # 点到直线距离公式
        distance = abs(a * cx - cy + b) / math.sqrt(a**2 + 1)
        return distance <= r

def _rectangle_rectangle_collide(r1, r2) -> bool:
    """矩形与矩形碰撞检测"""
    c1 = r1.get_corners()
    c2 = r2.get_corners()
    
    # 检查r1的每条边和r2的每条边是否相交
    for i in range(4):
        p1, p2 = c1[i], c1[(i+1)%4]
        for j in range(4):
            q1, q2 = c2[j], c2[(j+1)%4]
            if _segments_intersect(p1, p2, q1, q2):
                return True
    return False

def _segments_intersect(p1: list, p2: list, q1: list, q2: list) -> bool:
    """判断两条线段是否相交"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def _circle_rectangle_collide(circle, rect) -> bool:
    """圆与矩形碰撞检测"""
    corners = rect.get_corners()
    
    # 检查圆心是否在矩形内
    if _point_in_rectangle(circle.center, rect):
        return True
    
    # 检查圆是否与矩形的任何边相交
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        if _point_segment_distance(circle.center, p1, p2) <= circle.radius:
            return True
    return False

def _point_in_rectangle(point: list, rect) -> bool:
    """判断点是否在矩形内（考虑旋转）"""
    px, py = point
    corners = rect.get_corners()
    
    # 使用射线投射法
    intersections = 0
    test_ray = Ray(origin=[px, py], angle=0)
    
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        if _ray_segment_collide(test_ray, p1, p2):
            intersections += 1
    
    return intersections % 2 == 1

def _point_segment_distance(point: list, seg_start: list, seg_end: list) -> float:
    """计算点到线段的最短距离"""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # 向量法计算点到线段距离
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # 线段退化为点
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # 投影参数
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # 限制在[0,1]范围内
    
    # 最近点坐标
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    
    # 计算距离
    return math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

def _circle_circle_collide(c1: Circle, c2: Circle) -> bool:
    """圆与圆碰撞检测"""
    x0, y0 = c1.center
    r0 = c1.radius
    x1, y1 = c2.center
    r1 = c2.radius

    dx = x1 - x0
    dy = y1 - y0
    distance = math.sqrt(dx*dx + dy*dy)
    
    # 两圆相交或包含关系都算碰撞
    return distance <= (r0 + r1)

from typing import List, Union

# 新增 Ray 支持
def intersection(obj1: Union[Rectangle, Circle, Line, Ray],
                 obj2: Union[Rectangle, Circle, Line, Ray]) -> List[List[float]]:

    # ---------------- Line / Ray ----------------
    if isinstance(obj1, Ray) and isinstance(obj2, Line):
        return ray_line_intersection(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Line):
        return ray_line_intersection(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Circle):
        return ray_circle_intersection(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Circle):
        return ray_circle_intersection(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Rectangle):
        return ray_rectangle_intersection(obj1, obj2)
    if isinstance(obj2, Ray) and isinstance(obj1, Rectangle):
        return ray_rectangle_intersection(obj2, obj1)
    if isinstance(obj1, Ray) and isinstance(obj2, Ray):
        return ray_ray_intersection(obj1, obj2)

    # ---------------- Line / Circle / Rectangle ----------------
    if isinstance(obj1, Line) and isinstance(obj2, Line):
        return line_line_intersection(obj1, obj2)
    elif isinstance(obj1, Line) and isinstance(obj2, Rectangle):
        return rectangle_line_intersection(obj2, obj1)
    elif isinstance(obj2, Line) and isinstance(obj1, Rectangle):
        return rectangle_line_intersection(obj1, obj2)
    elif isinstance(obj1, Line) and isinstance(obj2, Circle):
        return circle_line_intersection(obj2, obj1)
    elif isinstance(obj2, Line) and isinstance(obj1, Circle):
        return circle_line_intersection(obj1, obj2)
    elif isinstance(obj1, Rectangle) and isinstance(obj2, Rectangle):
        return rectangle_rectangle_intersection(obj1, obj2)
    elif isinstance(obj1, Circle) and isinstance(obj2, Rectangle):
        return circle_rectangle_intersection(obj1, obj2)
    elif isinstance(obj1, Rectangle) and isinstance(obj2, Circle):
        return circle_rectangle_intersection(obj2, obj1)
    elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
        return circle_circle_intersection(obj1, obj2)

    print("Intersection not implemented for this type combination")
    return []

def ray_ray_intersection(ray1: Ray, ray2: Ray) -> List[list[float,float]]:
    """
    计算两条射线的交点
    Args:
        ray1: 第一条射线
        ray2: 第二条射线
    Returns:
        交点列表 [[x, y]]
    """
    x1, y1 = ray1.origin
    dx1, dy1 = ray1.dir
    x2, y2 = ray2.origin
    dx2, dy2 = ray2.dir

    # 解方程组:
    # x1 + t1 * dx1 = x2 + t2 * dx2
    # y1 + t1 * dy1 = y2 + t2 * dy2
    det = dx1 * (-dx2) - dy1 * (-dy2)
    
    if np.isclose(det, 0):
        return []  # 平行或重合
    
    # 计算参数t1和t2
    t1 = ((x2 - x1) * (-dx2) - (y2 - y1) * (-dy2)) / det
    t2 = (dx1 * (y2 - y1) - dy1 * (x2 - x1)) / det
    
    # 检查是否都在射线上(t >= 0)
    if t1 >= 0 and t2 >= 0:
        return [ray1.point_at(t1)]
    return []

class Map:
    def __init__(self):
        self.x_range=MAP_X_RANGE
        if(self.x_range[0]>self.x_range[1]):
            self.x_range[0],self.x_range[1]=self.x_range[1],self.x_range[0]
        self.y_range=MAP_Y_RANGE
        if(self.y_range[0]>self.y_range[1]):
            self.y_range[0],self.y_range[1]=self.y_range[1],self.y_range[0]
        self.entities=[]
        self.boundary=MAP_WITH_BOUNDARY
        self.target=Circle(center=MAP_TARGET_POSITION, radius=MAP_TARGET_RADIUS)
        config:list=MAP_ENTITIES_CONFIG
        if self.boundary:
            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            self.entities.append(Line(a=None, b=None, inf_k=True))  # x = x_min
            self.entities[-1].a = x_min
            self.entities.append(Line(a=None, b=None, inf_k=True))  # x = x_max
            self.entities[-1].a = x_max
            self.entities.append(Line(a=0, b=y_min))  # y = y_min
            self.entities.append(Line(a=0, b=y_max))  # y = y_max
        for item in config:
            t = item.get("type", "")
            if t == "Circle":
                c = Circle(center=item["center"], radius=item["radius"])
                self.entities.append(c)
            elif t == "Rectangle":
                r = Rectangle(
                    x=item["x"], y=item["y"],
                    width=item["width"], height=item["height"],
                    theta=item.get("theta", 0)
                )
                self.entities.append(r)
            elif t == "Line":
                l = Line(a=item.get("a", 1), b=item.get("b", 0), inf_k=item.get("inf_k", False))
                self.entities.append(l)
            elif t == "Ray":
                ray = Ray(origin=item["origin"], angle=item["angle"])
                self.entities.append(ray)
            else:
                print(f"Warning: Unknown type {t}, skipped.")
    def paint(self, **kwargs):
        """
        在已有 plt 上绘制整个地图
        每个 entity 调用自己的 paint 方法
        """
        ax = plt.gca()
        # 设置坐标范围为地图范围
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        for e in self.entities:
            try:
                e.paint(**kwargs)
            except AttributeError:
                print(f"Warning: entity {e} has no paint method")
        self.target.paint(color="green")
    
    def draw(self):
        for e in self.entities:
            e.draw(color="black",linewidth=1)
        self.target.draw(color="green")
