import numpy as np
from map import *
from param import *
from pilot import *

class SolidPhyState:
    def __init__(self,offset=(0,0)):
        self.offset_x,self.offset_y=offset
        #地面参考系
        self.x=0
        self.y=0
        self.theta=0
        
        self.vx=0 
        self.vy=0
        self.omiga=0
        
        self.ax=0
        self.ay=0
        self.alpha=0
    

        #imu测得物理量
        self.la=0 #纵、前向加速度
        self.ha=0 #侧、左向加速度
        self.imu_omiga = 0 # IMU测得角速度
        self.imu_alpha = 0 # IMU测得角加速度,逆时针

    def world2imu(self):
        """
        将地面参考系物理量转为imu测量物理量
        """
        dx,dy=(self.offset_x,self.offset_y)
        self.imu_omiga = self.omiga
        self.imu_alpha = self.alpha
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        
        la_pure = self.ax * cos_t + self.ay * sin_t
        ha_pure = -self.ax * sin_t + self.ay * cos_t

        

        accel_offset_x = -self.omiga**2 * dx - self.alpha * dy
        accel_offset_y = -self.omiga**2 * dy + self.alpha * dx

        self.la = la_pure + accel_offset_x
        self.ha = ha_pure + accel_offset_y

    def imu2world(self):
        """
        将imu测量物理量转为地面参考系物理量
        """
        dx,dy=(self.offset_x,self.offset_y)
        self.omiga = self.imu_omiga
        self.alpha = self.imu_alpha

        
        accel_offset_x = -self.omiga**2 * dx - self.alpha * dy
        accel_offset_y = -self.omiga**2 * dy + self.alpha * dx
        
        la_center = self.la - accel_offset_x
        ha_center = self.ha - accel_offset_y
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)

        self.ax = la_center * cos_t - ha_center * sin_t
        self.ay = la_center * sin_t + ha_center * cos_t
    
    def imu_output(self)->list[float]:
        return [self.la,self.ha,self.imu_omiga]
    
    def imu_output_with_noise(self)->list[float]:
        return [self.la+np.random.randn()*DRONE_LA_NOISE_VAR,
                self.ha+np.random.randn()*DRONE_HA_NOISE_VAR,
                self.imu_omiga+np.random.randn()*DRONE_IMU_OMIGA_NOISE_VAR]
    
    def set_pos(self,x:float,y:float,theta:float):
        """
        设置位置（绝对），高阶导数置0
        """
        self.x=x
        self.y=y
        self.theta=theta
        self.vx=0
        self.vy=0
        self.vz=0
        self.ax=0
        self.ay=0
        self.alpha=0
        self.world2imu()

    def set_speed(self,vx:float,vy:float,omiga:float):
        """
        设置速度（绝对），高阶导置0
        """
        self.vx=vx
        self.vy=vy
        self.omiga=omiga
        self.ax=0
        self.ay=0
        self.alpha=0
        self.world2imu()

    def set_acc(self,ax:float,ay:float,alpha:float):
        """
        设置加速度
        """
        self.ax=ax
        self.ay=ay
        self.alpha=alpha
        self.world2imu()
    
    def update(self, al: float, ah: float, imu_alpha: float, dt: float = SIMULATOR_DT):
        """
        RK4 (translation) + analytic rotation update
        al, ah       : IMU测得前向/左向加速度
        imu_alpha    : IMU测得角加速度（逆时针为正）
        dt           : 时间步长
        """

        dx, dy = self.offset_x, self.offset_y

        omiga0 = self.omiga
        theta0 = self.theta

        self.alpha = imu_alpha
        self.omiga = omiga0 + self.alpha * dt
        self.theta = theta0 + omiga0 * dt + 0.5 * self.alpha * dt * dt
        self.theta = self.normalize_angle(self.theta)

        la_center = al - (-self.omiga**2 * dx - self.alpha * dy)
        ha_center = ah - (-self.omiga**2 * dy + self.alpha * dx)
        def f(state, theta):
            """
            state = [x, y, vx, vy]
            """
            x, y, vx, vy = state
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            ax = la_center * cos_t - ha_center * sin_t
            ay = la_center * sin_t + ha_center * cos_t

            return np.array([vx, vy, ax, ay])

        state0 = np.array([self.x, self.y, self.vx, self.vy])

        theta_mid = self.normalize_angle(
            theta0 + 0.5 * (omiga0 + self.omiga) * dt
        )

        k1 = f(state0, theta0)
        k2 = f(state0 + 0.5 * dt * k1, theta_mid)
        k3 = f(state0 + 0.5 * dt * k2, theta_mid)
        k4 = f(state0 + dt * k3, self.theta)

        state_new = state0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.x, self.y, self.vx, self.vy = state_new.tolist()

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        self.ax = la_center * cos_t - ha_center * sin_t
        self.ay = la_center * sin_t + ha_center * cos_t


        self.la = al
        self.ha = ah
        self.imu_alpha = imu_alpha
    
    @staticmethod
    def normalize_angle(angle:float)->float:
        return np.arctan2(np.sin(angle), np.cos(angle))
        

class Ranger:
    def __init__(self,map:Map):
        self.x=0
        self.y=0
        self.theta=0
        self.rays=[]
        for _ in range(RANGER_SAMPLES):
            tmp=Ray([0,0],0)
            self.rays.append(tmp)
        self.fov=RANGER_FOV
        self.samples=RANGER_SAMPLES
        self.range=RANGER_VALID_DISTANCE
        self.ray_is_updated=False
        self.distance=[]
        self.map=map
    def update_rays(self):
        angles = np.linspace(self.theta - self.fov/2, self.theta + self.fov/2, self.samples)
        for i,angle in enumerate(angles):
            self.rays[i].set_pos(origin=[self.x, self.y], angle=angle)
    def set_pos(self,x:float=None,y:float=None,theta:float=None):
        if x is not None:
            self.x=x
        if y is not None:
            self.y=y
        if theta is not None:
            self.theta=theta
        self.ray_is_updated=False
    
    def scan(self):
        map=self.map
        if self.ray_is_updated==False:
            self.update_rays()
            self.ray_is_updated=True
        
        distances = []
        
        # 对每条射线进行扫描
        for ray in self.rays:
            min_distance = RANGER_INVALID_DISTANCE_MARK
            valid_points = []
            
            # 检查与地图中所有实体的交点
            for entity in map.entities:
                # 使用intersection函数计算射线与实体的交点
                points = intersection(ray, entity)
                
                # 计算每个交点到射线起点的距离
                for point in points:
                    distance = np.sqrt((point[0] - ray.origin[0])**2 + (point[1] - ray.origin[1])**2)
                    # 检查距离是否在有效范围内
                    if RANGER_VALID_DISTANCE[0] <= distance <= RANGER_VALID_DISTANCE[1]:
                        valid_points.append(distance)
            
            # 找到最近的有效交点距离
            if valid_points:
                min_distance = min(valid_points)
            
            distances.append(float(min_distance))
        
        self.distance=distances
    
    def paint(self, **kwargs):
        """
        绘制激光雷达的所有射线
        Args:
            ax: matplotlib轴对象，如果为None则使用当前轴
            **kwargs: 传递给plot的额外参数
        """ 
        if not self.rays or not self.ray_is_updated:
            self.update_rays()
            self.ray_is_updated = True
        if len(self.distance)==0:
            self.scan()

        max_range = RANGER_VALID_DISTANCE[1]
        
        for i, ray in enumerate(self.rays):
            
            # 如果有scan结果，使用实际探测距离
            if hasattr(self, 'distance') and len(self.distance) > i:
                dist = self.distance[i]
                if dist != RANGER_INVALID_DISTANCE_MARK:
                    end_x = self.x + dist * np.cos(ray.angle)
                    end_y = self.y + dist * np.sin(ray.angle)
                    # 探测到物体用蓝色
                    ray.paint(length=dist,linestyle='-',color="blue",alpha=0.7,linewidth=0.5)
                else:
                    ray.paint(length=max_range,linestyle='--',color="red",alpha=0.7,linewidth=0.5)
            else:
                ray.paint(length=max_range,linestyle='--',color="blue",alpha=0.7,linewidth=0.5)
    def draw(self):
        """
        绘制激光雷达的所有射线
        """ 
        if not self.rays or not self.ray_is_updated:
            self.update_rays()
            self.ray_is_updated = True
        if len(self.distance)==0:
            self.scan()

        max_range = RANGER_VALID_DISTANCE[1]
        
        for i, ray in enumerate(self.rays):
            
            # 如果有scan结果，使用实际探测距离
            if hasattr(self, 'distance') and len(self.distance) > i:
                dist = self.distance[i]
                if dist != RANGER_INVALID_DISTANCE_MARK:
                    # 探测到物体用蓝色
                    ray.draw(length=dist,linestyle='-',color="blue",alpha=0.7,linewidth=0.5,active=True)
                else:
                    ray.draw(length=max_range,linestyle='--',color="red",alpha=0.7,linewidth=0.5,active=True)
            else:
                ray.draw(length=max_range,linestyle='--',color="blue",alpha=0.7,linewidth=0.5,active=True)
        pass
class Drone:
    def __init__(self,name:str,ranger:Ranger):
        self.state=SolidPhyState()
        self.ranger=ranger
        self.shape=Circle(radius=DRONE_RADIUS)
        self.pilot=Pilot()
        #bool
        self.paint_ranger=DRONE_PAINT_RANGER_RAYS
        self.is_syn=False
        self.artist_no_arrow=len(MONITOR_MAP_GLOBAL_ALL_ARTISTS)
        graph,=plt.plot([],[],color="black")
        MONITOR_MAP_GLOBAL_ALL_ARTISTS.append(graph)
        
        
        
    def update(self, time:float):
        #感知
        self.ranger.update_rays()
        self.ranger.scan()
        imu_data=self.state.imu_output_with_noise()
        ranger_date=self.ranger.distance
        #决策
        order=self.pilot.get_order(imu=imu_data,ranger=ranger_date,time=time)
        #执行
        self.state.update(*order)
        self.syn_pos()
        
        
    def set_pos(self,x:float,y:float,theta:float):
        self.state.set_pos(x=x,y=y,theta=theta)
        self.shape.set_pos(center=(x,y))
        self.ranger.set_pos(x=x,y=y,theta=theta)
        self.is_syn=True

    def syn_pos(self):
        x=self.state.x
        y=self.state.y
        theta=self.state.theta
        self.shape.set_pos(center=(x,y))
        self.ranger.set_pos(x=x,y=y,theta=theta)
        self.is_syn=True
    def paint(self):
        if not self.is_syn:
            self.syn_pos()
            
        x = self.state.x
        y = self.state.y
        theta = self.state.theta

        self.shape.paint(color="red",linewidth=0.5)

        arrow_len = DRONE_RADIUS * 1.5
        head_x = x + arrow_len * np.cos(theta)
        head_y = y + arrow_len * np.sin(theta)
        
        plt.plot([x, head_x], [y, head_y], color='black', linewidth=1.5)

        if self.paint_ranger:
            self.ranger.paint()
    def draw(self):
        """
        动态更新模式：调用各组件的 draw 方法更新 Line2D 数据
        """
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        if not self.is_syn:
            self.syn_pos()
            
        x = self.state.x
        y = self.state.y
        theta = self.state.theta

        self.shape.set_pos(center=[x, y])
        self.shape.draw(color="red",linewidth="0.5",active=True)
        arrow_len = DRONE_RADIUS * 1.5
        head_x = x + arrow_len * np.cos(theta)
        head_y = y + arrow_len * np.sin(theta)
        

        arrow_artist = MONITOR_MAP_GLOBAL_ALL_ARTISTS[self.artist_no_arrow]
        arrow_artist.set_data([x, head_x], [y, head_y])
        MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.append(self.artist_no_arrow)


        if self.paint_ranger:
            self.ranger.draw()
        return MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
            
            

        
        



        




        
        
        
        
                
