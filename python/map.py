import numpy as np
from typing import List, Union
import math
import matplotlib.pyplot as plt
from param import *
class Circle:
    def __init__(self,center:list[float,float]=[0,0],radius:float=1):
      self.center=center
      self.radius=radius
    def set_pos(self,center:list[float,float]=None,radius:float=None):
      if center!=None:
        self.center=center 
      if radius!=None:
        self.radius=radius
    def paint(self, **kwargs):
        """绘制圆"""
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        plt.plot(x, y, **kwargs)



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

class Ray:
    def __init__(self, origin: list[float,float], angle: float):
        """
        :param origin: 射线起点 [x0, y0]
        :param angle: 射线方向角（弧度，0指向x正方向，逆时针为正）
        """
        self.origin = origin
        self.angle = angle
        self.dir = [math.cos(angle), math.sin(angle)]  # 单位方向向量

    def point_at(self, t: float) -> list[float,float]:
        """射线上 t 位置的点"""
        x0, y0 = self.origin
        dx, dy = self.dir
        return [x0 + t * dx, y0 + t * dy]
    
    def paint(self, length=PRINT_RAY_LENGTH_DEFAULT, **kwargs):
        """绘制射线，默认长度10"""
        end = self.point_at(length)
        plt.plot([self.origin[0], end[0]], [self.origin[1], end[1]], **kwargs)
        

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
def circle_rectangle_intersection(circle, rect) -> List[list[float,float]]:
    points = []
    corners = rect.get_corners()
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        temp_line = Line()
        temp_line.set_with_point(p1, p2)
        pts = circle_line_intersection(circle, temp_line)
        points.extend(pts)
    if points:
        points = np.unique(np.array(points), axis=0).tolist()
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
    def gen_map(self,config:list=MAP_ENTITIES_CONFIG):
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




