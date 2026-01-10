import matplotlib.pyplot as plt
import numpy as np
from map import Map
from drone import Drone

def test_ranger_paint():
    """测试Ranger.paint方法"""
    
    # 创建地图和无人机
    map_instance = Map()
    map_instance.gen_map()
    
    drone = Drone("test_drone", "sim")
    ranger = drone.ranger
    
    # 设置激光雷达位置和方向
    ranger.set_pos(x=6.0, y=0.0)
    ranger.theta = np.pi/4  # 45度
    
    # 执行扫描获取距离数据
    distances = ranger.scan(map_instance)
    ranger.distance = distances  # 将scan结果保存到ranger对象中
    
    # 绘制地图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置坐标范围
    ax.set_xlim(map_instance.x_range)
    ax.set_ylim(map_instance.y_range)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.set_title('Lidar Scan Visualization with Ranger.paint()')
    
    # 绘制地图实体
    for entity in map_instance.entities:
        try:
            if hasattr(entity, 'get_corners'):
                corners = entity.get_corners() + [entity.get_corners()[0]]
                x, y = zip(*corners)
                ax.plot(x, y, 'k-', linewidth=2)
            elif hasattr(entity, 'center'):
                theta = np.linspace(0, 2*np.pi, 100)
                x = entity.center[0] + entity.radius * np.cos(theta)
                y = entity.center[1] + entity.radius * np.sin(theta)
                ax.plot(x, y, 'k-', linewidth=2)
        except Exception as e:
            print(f"Warning: could not plot entity {entity}: {e}")
    
    # 使用Ranger的paint方法绘制射线
    ranger.paint(ax=ax)
    
    # 添加图例
    ax.legend()
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    test_ranger_paint()