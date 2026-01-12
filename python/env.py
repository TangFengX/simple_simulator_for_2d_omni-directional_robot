from map import *
from param import *
from pilot import *
from drone import *
import os
from datetime import datetime






class Simulator:
    def __init__(self):
        self.map=Map()
        ranger_inst=Ranger(map=self.map)
        self.drone=Drone(name="drone",ranger=ranger_inst)
        self.state="INIT"#"INIT" "READY" "RUN" "CLASH" "FINISH" "ERROR"
        self.log=""
        self.log_path=SIMULATOR_LOG_PATH
        self.time=0
    def env_init(self):
        """
        初始化仿真环境
        """
        #初始机体位置
        self.drone.set_pos(DRONE_X_INITIAL,DRONE_Y_INITIAL,DRONE_THETA_INITIAL)
        self.time=0
        self.state="READY"
        self.env_log()
        self.env_moniter_init()
        
        
        
    
    def env_start(self):
        if(self.state!="READY"):
            self.state=="ERROR"
            self.env_log("env uninitialized")

            return
        self.state="RUN"
        self.env_log()
        
    def env_run(self):
        if self.state!="RUN":
            self.state=="ERROR"
            self.env_log("calling env_run() when state!=RUN")
            return
        #机体状态更新
        self.drone.update(time=self.time)
        #时间更新
        self.time=self.time+SIMULATOR_DT
        #碰撞检测
        self.env_log()
        for entity in self.map.entities:
            if is_colliding(self.drone.shape,entity):
                self.state="CLASH"
                self.env_log("hit obstacle")

        #到达目标检测
        if is_colliding(self.drone.shape,self.map.target) and self.state=="RUN":
            self.state="REACH"
            self.env_log("reach")
        MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX=self.drone.draw() #由于这个全局变量可能和局部变量发生了一些冲突，导致这里不得不采取这种不安全的写法
    def main(self):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX=[]
        if self.time>SIMULATOR_MAX_TIME and SIMULATOR_MAX_TIME>0:
            self.state="ERROR"
            self.env_log("time exceed")
        if self.state=="INIT":
            self.env_init()
        elif self.state=="READY":
            self.env_start()
        elif self.state=="RUN":
            self.env_run()
        elif self.state=="REACH" or self.state=="CLASH" or self.state=="ERROR":
            self.env_log_dump(self.state)
            exit(0)
        else:
            self.env_log("unkonw state")
            self.env_log_dump(self.state)
            assert 0 
        
    def env_log(self,text:str=""):
        tmp=""
        tmp+=f"[{self.state}]"
        tmp+=f" t={self.time:.3f}"
        tmp+=f" {text}\n"
        if(self.state=="RUN"):
            tmp+=f"\tx:{self.drone.state.x:.2f} y:{self.drone.state.y:.2f} theta/pi:{(self.drone.state.theta/np.pi):.2f}\n"
            tmp+=f"\tvx:{self.drone.state.vx:.2f} vy:{self.drone.state.vy:.2f} omiga/pi:{(self.drone.state.omiga/np.pi):.2f}\n"
            tmp+=f"\tax:{self.drone.state.ax:.2f} ay:{self.drone.state.ay:.2f} alpha/pi:{(self.drone.state.alpha/np.pi):.2f}\n"
            tmp+=f"\tdistance:{self.drone.ranger.distance}\n"
        if SIMULATOR_LOG_OUTPUT_TERMINAL:
            print(tmp)
        self.log+=tmp
            

    def env_log_dump(self,last_state="NONE"):
        path=self.log_path
        if not os.path.exists(path):
            os.makedirs(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{last_state}.log"
        full_path = os.path.join(path, file_name)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                    f.write(str(self.log))
            print(f"日志已成功保存至: {full_path}")
        except Exception as e:
            print(f"写入日志失败: {e}")
        
    def env_moniter_init(self):
        MONITOR_MAP_AX.set_xlim(*MAP_X_RANGE)
        MONITOR_MAP_AX.set_ylim(*MAP_Y_RANGE)
        MONITOR_MAP_AX.set_aspect("equal","box")
        MONITOR_MAP_AX.grid(True)
        self.map.draw()
        
    
    def env_monitor_update_render_list(self):
        global MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX
        l=MONITOR_MAP_GLOBAL_ALL_DYNAMIC_ARTISTS_INDEX.copy()
        return [MONITOR_MAP_GLOBAL_ALL_ARTISTS[i] for i in l]

    def update(self,frames):
        for _ in range(RATIO_OF_PHYSCAL_FRAME_TO_MONITER_FRAME):
            self.main()
        return self.env_monitor_update_render_list()